import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import hstack

# ============================================================
# CREAR APP (SOLO UNA VEZ)
# ============================================================

app = FastAPI(title="Spotify Popularity Predictor")

# ✅ CORS (IMPORTANTE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CARGAR MODELOS
# ============================================================

print("Cargando modelos...")

xgb_model = joblib.load('xgb.pkl')
lgb_model = joblib.load('lgb.pkl')
rf_model = joblib.load('rf.pkl')

pre = joblib.load('preprocessor.pkl')
tfidf = joblib.load('tfidf.pkl')
meta = joblib.load('meta_model.pkl')

print("Modelos cargados")

# ============================================================
# INPUT
# ============================================================

class SongInput(BaseModel):
    artists: str
    album_name: str
    track_name: str
    track_genre: str
    duration_ms: float
    energy: float
    danceability: float

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
# ENDPOINT
# ============================================================

@app.post("/predict")
def predict(song: SongInput):

    df = pd.DataFrame([song.dict()])
    df = crear_features(df)

    # columnas necesarias
    df['artists_te'] = 0
    df['track_id'] = 0

    # TF-IDF (ANTES de modificar columnas)
    X_tfidf = tfidf.transform(df['track_name'])

    # asegurar columnas esperadas
    expected_cols = pre.feature_names_in_

    for col in expected_cols:
      if col not in df.columns:
        df[col] = 0

    df = df[expected_cols]

    # tabular
    X_prep = pre.transform(df)

    # merge
    X_final = hstack([X_prep, X_tfidf])

    # modelos
    pred_xgb = xgb_model.predict(X_final)
    pred_lgb = lgb_model.predict(X_final)
    pred_rf = rf_model.predict(X_final)

    # stacking
    X_meta = np.column_stack([pred_xgb, pred_lgb, pred_rf])
    final_pred = meta.predict(X_meta)

    final_pred = float(np.clip(final_pred, 0, 100)[0])

    return {"predicted_popularity": final_pred}