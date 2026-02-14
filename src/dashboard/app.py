import os
import sqlite3
import joblib
import pandas as pd
import streamlit as st

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "db", "ridewise.sqlite")

MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.joblib")
st.caption(f"DB_PATH: {DB_PATH}")

st.set_page_config(page_title="RideWise Churn Dashboard", layout="wide")

st.title("🚕 RideWise Customer Churn Dashboard")

# --- Load data ---
@st.cache_data
def load_data():
    @st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at: {DB_PATH}")
        st.info("Fix: push db/ridewise.sqlite to GitHub, or rebuild DB on the server.")
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM user_features", conn)
    metrics = pd.read_sql("SELECT * FROM model_metrics", conn)
    conn.close()
    return df, metrics

df, metrics = load_data()

# --- KPIs ---
col1, col2, col3 = st.columns(3)

total_users = len(df)
churn_rate = df["churn_30d"].mean()

col1.metric("Total Users", f"{total_users:,}")
col2.metric("Churn Rate", f"{churn_rate:.2%}")
col3.metric("Avg Trips", f"{df['trips_count'].mean():.1f}")

st.divider()

# --- Segment Distribution ---
st.subheader("User Segments")
seg_counts = df["segment"].value_counts()
st.bar_chart(seg_counts)

# --- Churn Distribution ---
st.subheader("Churn Distribution")
st.bar_chart(df["churn_30d"].value_counts())

st.divider()

# --- Model Metrics ---
st.subheader("Model Performance")
st.dataframe(metrics)

st.divider()

# --- Prediction Tool ---
st.subheader("Predict User Churn")

user_id = st.selectbox("Select user_id", df["user_id"].head(500))

if st.button("Predict"):
    model = joblib.load(MODEL_PATH)

    row = df[df["user_id"] == user_id].iloc[0]

    drop_cols = ["user_id", "signup_date", "last_trip_time", "first_trip_time", "last_session_time", "churn_30d", "segment"]
    X = row.drop(labels=[c for c in drop_cols if c in row.index])
    X_df = pd.DataFrame([X])

    proba = model.predict_proba(X_df)[:, 1][0]

    if proba >= 0.7:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    st.success(f"Churn Probability: {proba:.2%}")
    st.write(f"Risk Band: **{risk}**")
