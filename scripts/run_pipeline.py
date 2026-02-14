import sys
import os

# --- Make src importable ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

import sqlite3
import pandas as pd

from src.data.load_data import load_all
from src.features.build_features import build_user_features
from src.features.segmentation import assign_segment
from src.models.train import train_models


def main():
    DATA_DIR = os.path.join(BASE_DIR, "data", "data_ridewise")
    DB_PATH = os.path.join(BASE_DIR, "db", "ridewise.sqlite")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data from:", DATA_DIR)
    riders, trips, sessions, promotions, drivers = load_all(DATA_DIR)

    print("Building features...")
    feats = build_user_features(riders, trips, sessions)

    print("Assigning segments...")
    feats = assign_segment(feats)

    print("Training models...")
    results = train_models(feats, MODELS_DIR)
    print("Training results:", results)

    print("Saving to SQLite:", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    feats.to_sql("user_features", conn, if_exists="replace", index=False)

    metrics_df = pd.DataFrame([{"model": k, **v} for k, v in results.items()])
    metrics_df.to_sql("model_metrics", conn, if_exists="replace", index=False)
    conn.close()

    print("Done.")
    print("DB:", DB_PATH)
    print("Models saved in:", MODELS_DIR)


if __name__ == "__main__":
    main()
