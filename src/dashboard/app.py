import os
import sqlite3
import pandas as pd
import streamlit as st

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "ridewise.sqlite")

st.title("🚕 RideWise Customer Churn Dashboard")
st.caption(f"DB_PATH: {DB_PATH}")

# --- Load data ---
@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at: {DB_PATH}")
        st.stop()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM user_features", conn)
    metrics = pd.read_sql("SELECT * FROM model_metrics", conn)
    conn.close()
    return df, metrics


df, metrics = load_data()

# --- Metrics ---
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Users", len(df))

with col2:
    churn_rate = df["churn_30d"].mean() if "churn_30d" in df.columns else 0
    st.metric("Churn Rate", f"{churn_rate:.2%}")

# --- Segment distribution ---
if "segment" in df.columns:
    st.subheader("Customer Segments")
    seg_counts = df["segment"].value_counts()
    st.bar_chart(seg_counts)

# --- Churn distribution ---
if "churn_30d" in df.columns:
    st.subheader("Churn Distribution")
    churn_counts = df["churn_30d"].value_counts()
    st.bar_chart(churn_counts)

# --- Preview ---
st.subheader("Sample Users")
st.dataframe(df.head(20))
