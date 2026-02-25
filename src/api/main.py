import os
import json
import sqlite3
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# -----------------------------
# Paths / Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "ridewise.sqlite")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.joblib")

# Tuned threshold from training
RF_THRESHOLD = 0.25

app = FastAPI(title="RideWise API", version="1.0")


# -----------------------------
# Schemas
# -----------------------------
class ChurnRequest(BaseModel):
    user_id: str


# -----------------------------
# Helpers
# -----------------------------
def _get_user_row(user_id: str):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found. Run pipeline first.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM user_features WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()

    return dict(row) if row else None


def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model not found. Run pipeline first.")
    return joblib.load(MODEL_PATH)


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/churn")
def predict_churn(req: ChurnRequest):
    row = _get_user_row(req.user_id)
    if not row:
        raise HTTPException(status_code=404, detail="user_id not found")

    model = _load_model()

    # Remove leakage + non-feature columns
    drop_cols = [
        "user_id",
        "signup_date",
        "last_trip_time",
        "first_trip_time",
        "last_session_time",
        "recency_days",
        "churn_30d",
        "segment",
    ]

    X = {k: v for k, v in row.items() if k not in drop_cols}
    X_df = pd.DataFrame([X])

    proba = float(model.predict_proba(X_df)[:, 1][0])

    threshold = RF_THRESHOLD
    predicted_churn = int(proba >= threshold)

    risk_band = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"

    return {
        "user_id": req.user_id,
        "churn_probability": proba,
        "threshold": threshold,
        "predicted_churn": predicted_churn,
        "risk_band": risk_band,
    }


@app.get("/dashboard/metrics")
def dashboard_metrics():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found. Run pipeline first.")

    conn = sqlite3.connect(DB_PATH)

    total_users = conn.execute("SELECT COUNT(*) FROM user_features").fetchone()[0]
    churn_rate = conn.execute("SELECT AVG(churn_30d) FROM user_features").fetchone()[0]

    metrics = pd.read_sql("SELECT * FROM model_metrics", conn).to_dict(orient="records")
    conn.close()

    return {
        "total_users": int(total_users),
        "estimated_churn_rate": float(churn_rate) if churn_rate is not None else None,
        "model_metrics": metrics,
        "model_name": "random_forest",
        "model_threshold": RF_THRESHOLD,
    }
