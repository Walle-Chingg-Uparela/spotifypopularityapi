import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib

from scipy.sparse import hstack


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


def predict():

    print("Cargando modelos...")

    xgb_model = joblib.load('xgb.pkl')
    lgb_model = joblib.load('lgb.pkl')
    rf_model = joblib.load('rf.pkl')

    pre = joblib.load('preprocessor.pkl')
    tfidf = joblib.load('tfidf.pkl')
    meta = joblib.load('meta_model.pkl')

    test_df = pd.read_csv(
        "https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTest_Spotify.csv",
        index_col=0
    )

    print("Creando features...")
    test_df = crear_features(test_df)

    # 🔥 IMPORTANTE: recrear columna
    test_df['artists_te'] = 0

    X_tfidf = tfidf.transform(test_df['track_name'])

    drop_cols = ['track_id', 'track_name']
    X_test = test_df.drop(columns=drop_cols, errors='ignore')

    X_prep = pre.transform(X_test)

    X_final = hstack([X_prep, X_tfidf])

    print("Prediciendo...")

    pred_xgb = xgb_model.predict(X_final)
    pred_lgb = lgb_model.predict(X_final)
    pred_rf = rf_model.predict(X_final)

    X_meta = np.column_stack([pred_xgb, pred_lgb, pred_rf])

    final_pred = meta.predict(X_meta)
    final_pred = np.clip(final_pred, 0, 100)

    submission = pd.DataFrame(final_pred, index=test_df.index, columns=['Popularity'])
    submission.to_csv('predictions.csv')

    print("\nPredicciones listas")
    print(submission.head())


if __name__ == "__main__":
    predict()