import os
import sqlite3
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "ridewise.sqlite")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(title="RideWise API", version="1.0")

class ChurnRequest(BaseModel):
    user_id: str


def _get_user_row(user_id: int):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found. Run the pipeline first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM user_features WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/churn")
def predict_churn(req: ChurnRequest):
    row = _get_user_row(req.user_id)
    if not row:
        raise HTTPException(status_code=404, detail="user_id not found")

    model_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model not found. Run the pipeline first.")

    model = joblib.load(model_path)

    # Build X row (must match training drop columns)
    drop_cols = ["user_id", "signup_date", "last_trip_time", "first_trip_time", "last_session_time", "churn_30d", "segment"]
    X = {k: v for k, v in row.items() if k not in drop_cols}
    X_df = pd.DataFrame([X])

    proba = float(model.predict_proba(X_df)[:, 1][0])
    risk = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"

    return {"user_id": req.user_id, "churn_probability": proba, "risk_band": risk}


@app.get("/dashboard/metrics")
def dashboard_metrics():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database not found. Run the pipeline first.")

    conn = sqlite3.connect(DB_PATH)
    total_users = conn.execute("SELECT COUNT(*) FROM user_features").fetchone()[0]
    churn_rate = conn.execute("SELECT AVG(churn_30d) FROM user_features").fetchone()[0]

    metrics = pd.read_sql("SELECT * FROM model_metrics", conn).to_dict(orient="records")
    conn.close()

    return {
        "total_users": int(total_users),
        "estimated_churn_rate": float(churn_rate) if churn_rate is not None else None,
        "model_metrics": metrics,
    }
