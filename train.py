
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import lightgbm as lgb


# ============================================================
# TARGET ENCODING
# ============================================================

def cv_target_encoding(train, test, col, target_col, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_enc = np.zeros(len(train))
    test_enc = np.zeros(len(test))
    global_mean = train[target_col].mean()

    for tr_idx, val_idx in kf.split(train):
        X_tr = train.iloc[tr_idx]
        X_val = train.iloc[val_idx]

        means = X_tr.groupby(col)[target_col].mean()

        train_enc[val_idx] = X_val[col].map(means).fillna(global_mean)
        test_enc += test[col].map(means).fillna(global_mean) / n_splits

    return train_enc, test_enc


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def crear_features(df):
    df = df.copy()

    for c in ['artists', 'album_name', 'track_name', 'track_genre']:
        df[c] = df.get(c, 'missing').fillna('missing').astype(str)

    df['artist_count'] = df['artists'].str.count(';') + 1
    df['track_name_len'] = df['track_name'].str.len()
    df['is_remix'] = df['track_name'].str.lower().str.contains('remix').astype(int)

    df['duration_ms'] = pd.to_numeric(df.get('duration_ms', 0), errors='coerce')
    df['duration_ms_log'] = np.log1p(df['duration_ms'].clip(lower=0))

    if {'energy', 'danceability'}.issubset(df.columns):
        df['energy_x_danceability'] = df['energy'] * df['danceability']

    return df


# ============================================================
# OOF MODEL
# ============================================================

def oof_model(model, X, y, X_test, name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof = np.zeros(len(y))
    test_pred = np.zeros(X_test.shape[0])

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)

        pred_va = model.predict(X_va)
        pred_te = model.predict(X_test)

        oof[va_idx] = pred_va
        test_pred += pred_te / 5

        rmse = np.sqrt(mean_squared_error(y_va, pred_va))
        print(f"{name} Fold {fold} RMSE: {rmse:.4f}")

    return oof, test_pred


# ============================================================
# TRAIN
# ============================================================

def train_model():

    print("Cargando datos...")

    train_df = pd.read_csv(
        "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTrain_Spotify.csv",
        index_col=0
    )

    test_df = pd.read_csv(
        "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTest_Spotify.csv",
        index_col=0
    )

    target = 'popularity'

    # ---------- FEATURES ----------
    train_df = crear_features(train_df)
    test_df = crear_features(test_df)

    # ---------- TARGET ENCODING ----------
    train_df['artists_te'], test_df['artists_te'] = cv_target_encoding(
        train_df, test_df, 'artists', target
    )

    # ---------- TF-IDF ----------
    tfidf = TfidfVectorizer(max_features=100)
    X_tfidf_train = tfidf.fit_transform(train_df['track_name'])
    X_tfidf_test = tfidf.transform(test_df['track_name'])

    # ---------- TABULAR ----------
    drop_cols = [target, 'track_id', 'track_name']
    X = train_df.drop(columns=drop_cols, errors='ignore')
    y = train_df[target].values
    X_test = test_df.drop(columns=drop_cols, errors='ignore')

    cat_cols = ['track_genre']
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    X_prep = pre.fit_transform(X)
    X_test_prep = pre.transform(X_test)

    from scipy.sparse import hstack
    X_final = hstack([X_prep, X_tfidf_train])
    X_test_final = hstack([X_test_prep, X_tfidf_test])

    # ============================================================
    # MODELOS
    # ============================================================

    print("\n--- XGBOOST ---")
    xgb_model = xgb.XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=6, n_jobs=4)
    oof_xgb, _ = oof_model(xgb_model, X_final, y, X_test_final, "XGB")

    print("\n--- LIGHTGBM ---")
    lgb_model = lgb.LGBMRegressor(n_estimators=450, learning_rate=0.05, n_jobs=4)
    oof_lgb, _ = oof_model(lgb_model, X_final, y, X_test_final, "LGB")

    print("\n--- RANDOM FOREST ---")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=4)
    oof_rf, _ = oof_model(rf_model, X_final, y, X_test_final, "RF")

    # ============================================================
    # STACKING
    # ============================================================

    X_meta = np.column_stack([oof_xgb, oof_lgb, oof_rf])

    meta = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    meta.fit(X_meta, y)

    rmse = np.sqrt(mean_squared_error(y, meta.predict(X_meta)))
    print("\nRMSE FINAL:", rmse)

    # ============================================================
    # 🔥 REFIT FINAL (CLAVE)
    # ============================================================

    print("\nReentrenando modelos con TODOS los datos...")

    xgb_model.fit(X_final, y)
    lgb_model.fit(X_final, y)
    rf_model.fit(X_final, y)

    # ============================================================
    # GUARDAR
    # ============================================================

    joblib.dump(xgb_model, 'xgb.pkl')
    joblib.dump(lgb_model, 'lgb.pkl')
    joblib.dump(rf_model, 'rf.pkl')

    joblib.dump(pre, 'preprocessor.pkl')
    joblib.dump(tfidf, 'tfidf.pkl')
    joblib.dump(meta, 'meta_model.pkl')

    print("\nModelos guardados correctamente")


if __name__ == "__main__":
    train_model()