import os
import sqlite3
import pandas as pd
import streamlit as st
import requests

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "ridewise.sqlite")

# Local FastAPI (change if you deploy API)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RideWise Dashboard", page_icon="🚕", layout="wide")


# -----------------------------
# Data load
# -----------------------------
@st.cache_data
def load_data(db_path: str):
    if not os.path.exists(db_path):
        st.error("Database not found. Run pipeline first: py scripts/run_pipeline.py")
        st.stop()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM user_features", conn)
    metrics = pd.read_sql("SELECT * FROM model_metrics", conn)
    conn.close()
    return df, metrics


df, metrics = load_data(DB_PATH)

# -----------------------------
# Header
# -----------------------------
st.title("🚕 RideWise Customer Churn Dashboard")
st.caption("Explore churn, segments, and predict risk for individual users.")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔎 Filters")

# Segment filter
segments = ["All"] + sorted(df["segment"].dropna().unique().tolist()) if "segment" in df.columns else ["All"]
segment_choice = st.sidebar.selectbox("Customer Segment", segments, key="filter_segment")

# Churn filter
churn_choice = st.sidebar.selectbox(
    "Churn Status",
    ["All", "Churned (30d inactive)", "Active"],
    key="filter_churn",
)

# City filter (optional)
cities = ["All"] + sorted(df["city"].dropna().astype(str).unique().tolist()) if "city" in df.columns else ["All"]
city_choice = st.sidebar.selectbox("City", cities, key="filter_city")

# Loyalty filter (optional)
loyalty_vals = ["All"] + sorted(df["loyalty_status"].dropna().astype(str).unique().tolist()) if "loyalty_status" in df.columns else ["All"]
loyalty_choice = st.sidebar.selectbox("Loyalty Status", loyalty_vals, key="filter_loyalty")

# Age range (optional)
if "age" in df.columns and pd.api.types.is_numeric_dtype(df["age"]):
    age_min = int(df["age"].dropna().min()) if df["age"].dropna().size else 0
    age_max = int(df["age"].dropna().max()) if df["age"].dropna().size else 100
    age_range = st.sidebar.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max), key="filter_age")
else:
    age_range = None

# Trips range (optional)
if "trips_count" in df.columns and pd.api.types.is_numeric_dtype(df["trips_count"]):
    trips_min = int(df["trips_count"].dropna().min()) if df["trips_count"].dropna().size else 0
    trips_max = int(df["trips_count"].dropna().max()) if df["trips_count"].dropna().size else 0
    trips_range = st.sidebar.slider("Trips count range", min_value=trips_min, max_value=trips_max, value=(trips_min, trips_max), key="filter_trips")
else:
    trips_range = None

# Apply filters
filtered = df.copy()

if segment_choice != "All" and "segment" in filtered.columns:
    filtered = filtered[filtered["segment"] == segment_choice]

if "churn_30d" in filtered.columns:
    if churn_choice == "Churned (30d inactive)":
        filtered = filtered[filtered["churn_30d"] == 1]
    elif churn_choice == "Active":
        filtered = filtered[filtered["churn_30d"] == 0]

if city_choice != "All" and "city" in filtered.columns:
    filtered = filtered[filtered["city"].astype(str) == str(city_choice)]

if loyalty_choice != "All" and "loyalty_status" in filtered.columns:
    filtered = filtered[filtered["loyalty_status"].astype(str) == str(loyalty_choice)]

if age_range is not None and "age" in filtered.columns:
    filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]

if trips_range is not None and "trips_count" in filtered.columns:
    filtered = filtered[(filtered["trips_count"] >= trips_range[0]) & (filtered["trips_count"] <= trips_range[1])]

st.sidebar.caption(f"Filtered users: {len(filtered):,}")

# Download filtered data
st.sidebar.download_button(
    "⬇️ Download filtered data (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="ridewise_filtered_users.csv",
    mime="text/csv",
    key="download_filtered",
)

# -----------------------------
# KPIs (Filtered)
# -----------------------------
def safe_mean(series, default=0.0):
    try:
        return float(series.mean())
    except Exception:
        return default


churn_rate = safe_mean(filtered["churn_30d"], 0.0) if "churn_30d" in filtered.columns and len(filtered) else 0.0
avg_trips = safe_mean(filtered["trips_count"], 0.0) if "trips_count" in filtered.columns and len(filtered) else 0.0
avg_spend_trip = safe_mean(filtered["spend_per_trip"], 0.0) if "spend_per_trip" in filtered.columns and len(filtered) else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Users (filtered)", f"{len(filtered):,}")
k2.metric("Churn rate (filtered)", f"{churn_rate:.2%}")
k3.metric("Avg trips/user", f"{avg_trips:.2f}")
k4.metric("Avg spend/trip", f"{avg_spend_trip:.2f}")

st.divider()

# -----------------------------
# Model metrics
# -----------------------------
st.subheader("📌 Model Metrics")
st.dataframe(metrics, use_container_width=True)

# -----------------------------
# Charts
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    if "segment" in df.columns:
        st.subheader("Customer Segments (overall)")
        seg_counts = df["segment"].value_counts()
        st.bar_chart(seg_counts)

with c2:
    if "churn_30d" in filtered.columns:
        st.subheader("Churn Distribution (filtered)")
        churn_counts = filtered["churn_30d"].value_counts()
        st.bar_chart(churn_counts)

# Churn by segment
if "segment" in df.columns and "churn_30d" in df.columns:
    st.subheader("Churn Rate by Segment (overall)")
    churn_by_seg = df.groupby("segment")["churn_30d"].mean().sort_values(ascending=False)
    st.bar_chart(churn_by_seg)

# Top cities
if "city" in df.columns:
    st.subheader("Top Cities (overall)")
    top_cities = df["city"].astype(str).value_counts().head(10)
    st.bar_chart(top_cities)

st.divider()

# -----------------------------
# Preview table
# -----------------------------
st.subheader("👀 Sample Users (filtered)")
st.dataframe(filtered.head(30), use_container_width=True)

# -----------------------------
# Prediction helpers
# -----------------------------
def api_predict(user_id: str):
    res = requests.post(
        f"{API_URL}/predict/churn",
        json={"user_id": user_id},
        timeout=10,
    )
    if res.status_code != 200:
        return None
    return res.json()


# -----------------------------
# High-risk users section
# -----------------------------
st.divider()
st.subheader("🚨 Top High-Risk Users (from API)")

st.caption("This calls the FastAPI endpoint to compute churn probability. (Uses the tuned threshold.)")

top_n = st.slider("How many users to score?", min_value=5, max_value=50, value=15, step=5, key="topn_score")

if "user_id" not in filtered.columns or len(filtered) == 0:
    st.warning("No users available in the current filter.")
else:
    # Take a sample of user ids to score (for speed)
    candidate_ids = filtered["user_id"].astype(str).dropna().unique().tolist()
    candidate_ids = candidate_ids[: max(top_n * 3, top_n)]  # small buffer

    if st.button("Score users", key="score_users_btn"):
        rows = []
        with st.spinner("Scoring users via API..."):
            for uid in candidate_ids:
                out = api_predict(uid)
                if out:
                    rows.append(out)

        if rows:
            scored = pd.DataFrame(rows)
            scored["churn_probability"] = scored["churn_probability"].astype(float)
            scored = scored.sort_values("churn_probability", ascending=False).head(top_n)
            st.dataframe(scored, use_container_width=True)
        else:
            st.error("Could not score users. Make sure API is running: py -m uvicorn src.api.main:app --reload")

# -----------------------------
# Single prediction section
# -----------------------------
st.divider()
st.subheader("🔮 Customer Churn Prediction")

user_id_options = (
    filtered["user_id"].astype(str).dropna().unique()
    if "user_id" in filtered.columns
    else []
)
user_id_options = sorted(user_id_options)

if len(user_id_options) == 0:
    st.warning("No users available in the current filter. Change filters to see user IDs.")
else:
    selected_user_id = st.selectbox(
        "Choose a User ID",
        user_id_options,
        key="user_select",
    )

    manual_user_id = st.text_input(
        "Or type a User ID (optional)",
        "",
        key="manual_user_id_input",
    )

    user_to_predict = manual_user_id.strip() if manual_user_id.strip() else selected_user_id

    if st.button("Predict Churn", key="predict_btn"):
        try:
            data = api_predict(user_to_predict)

            if data is None:
                st.error("User not found (or API returned an error).")
            else:
                prob = float(data.get("churn_probability", 0))
                risk = data.get("risk_band", "Unknown")
                threshold = float(data.get("threshold", 0.5))
                predicted = int(data.get("predicted_churn", 0))

                m1, m2, m3 = st.columns(3)
                m1.metric("Churn Probability", f"{prob:.2%}")
                m2.metric("Threshold Used", f"{threshold:.2f}")
                m3.metric("Predicted Churn", "YES" if predicted == 1 else "NO")

                if risk == "High":
                    st.error(f"Risk Band: {risk}")
                elif risk == "Medium":
                    st.warning(f"Risk Band: {risk}")
                else:
                    st.info(f"Risk Band: {risk}")

        except Exception:
            st.error(f"Prediction API not running. Start it with: py -m uvicorn src.api.main:app --reload")
