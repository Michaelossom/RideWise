import os
import time
import sqlite3
import pandas as pd
import streamlit as st
import requests

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="RideWise Dashboard", page_icon="🚕", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "ridewise.sqlite")

# Use env var if set (best practice for deployment), else default to your Render API
API_URL = os.getenv("API_URL", "https://ridewise-api.onrender.com").rstrip("/")

# -----------------------------
# Helpers
# -----------------------------
def api_healthcheck() -> bool:
    """Wake Render free instance / verify API is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=30)
        return r.status_code == 200
    except Exception:
        return False


def api_predict(user_id: str, retries: int = 4):
    """Predict churn for user_id with wake + retries (Render free can sleep)."""
    # Wake the API (non-blocking)
    api_healthcheck()

    last_err = None
    for attempt in range(retries):
        try:
            res = requests.post(
                f"{API_URL}/predict/churn",
                json={"user_id": str(user_id)},
                timeout=30,
            )
            if res.status_code == 200:
                return res.json()

            # capture message if FastAPI returns one
            last_err = f"HTTP {res.status_code}: {res.text}"

        except Exception as e:
            last_err = str(e)

        time.sleep(2 + attempt)  # backoff

    return {"error": last_err or "Unknown error"}


@st.cache_data
def load_data(db_path: str):
    """Load DB if present; return (df, metrics)."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM user_features", conn)
    metrics = pd.read_sql("SELECT * FROM model_metrics", conn)
    conn.close()
    return df, metrics


def safe_mean(series, default=0.0):
    try:
        return float(series.mean())
    except Exception:
        return default


# -----------------------------
# Header
# -----------------------------
st.title("🚕 RideWise Customer Churn Dashboard")
st.caption("Explore churn, segments, and predict risk for individual users.")

with st.expander("⚙️ Connection Settings", expanded=False):
    st.write("**API_URL**:", API_URL)
    ok = api_healthcheck()
    st.write("**API status**:", "✅ Reachable" if ok else "⚠️ Not reachable (might be sleeping)")

# -----------------------------
# Try DB load (optional)
# -----------------------------
db_available = os.path.exists(DB_PATH)

if db_available:
    df, metrics = load_data(DB_PATH)
else:
    df = pd.DataFrame()
    metrics = pd.DataFrame()
    st.warning(
        "Database file not found in this environment. "
        "Dashboard will run in **API-only mode** (prediction still works)."
    )

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔎 Filters")

if not df.empty:
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
else:
    # API-only mode: minimal sidebar
    st.sidebar.info("API-only mode (no DB loaded).")

# -----------------------------
# Main content (if DB available)
# -----------------------------
if not df.empty:
    churn_rate = safe_mean(filtered["churn_30d"], 0.0) if "churn_30d" in filtered.columns and len(filtered) else 0.0
    avg_trips = safe_mean(filtered["trips_count"], 0.0) if "trips_count" in filtered.columns and len(filtered) else 0.0
    avg_spend_trip = safe_mean(filtered["spend_per_trip"], 0.0) if "spend_per_trip" in filtered.columns and len(filtered) else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Users (filtered)", f"{len(filtered):,}")
    k2.metric("Churn rate (filtered)", f"{churn_rate:.2%}")
    k3.metric("Avg trips/user", f"{avg_trips:.2f}")
    k4.metric("Avg spend/trip", f"{avg_spend_trip:.2f}")

    st.divider()

    st.subheader("📌 Model Metrics")
    st.dataframe(metrics, use_container_width=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if "segment" in df.columns:
            st.subheader("Customer Segments (overall)")
            st.bar_chart(df["segment"].value_counts())

    with c2:
        if "churn_30d" in filtered.columns:
            st.subheader("Churn Distribution (filtered)")
            st.bar_chart(filtered["churn_30d"].value_counts())

    if "segment" in df.columns and "churn_30d" in df.columns:
        st.subheader("Churn Rate by Segment (overall)")
        churn_by_seg = df.groupby("segment")["churn_30d"].mean().sort_values(ascending=False)
        st.bar_chart(churn_by_seg)

    if "city" in df.columns:
        st.subheader("Top Cities (overall)")
        st.bar_chart(df["city"].astype(str).value_counts().head(10))

    st.divider()
    st.subheader("👀 Sample Users (filtered)")
    st.dataframe(filtered.head(30), use_container_width=True)
# -----------------------------
# Feature Importance (Model Explainability)
# -----------------------------
st.divider()
st.subheader("🧠 Feature Importance (Random Forest)")

MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.joblib")

if not os.path.exists(MODEL_PATH):
    st.warning(
        "Model file not found locally in this environment. "
        "If you're deployed, ensure models/*.joblib are included in the repo."
    )
else:
    try:
        pipe = joblib.load(MODEL_PATH)

        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]

        feature_names = pre.get_feature_names_out()
        importances = clf.feature_importances_

        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        top_n = st.slider("Top N features", 5, 50, 15, 5, key="fi_topn")
        st.caption("These importances come from the trained Random Forest model.")

        st.bar_chart(fi.head(top_n).set_index("feature")["importance"])

        st.subheader("📈 Importance Distribution")
        st.line_chart(
            np.sort(fi["importance"].values)[::-1]
        )

        with st.expander("See full importance table"):
            st.dataframe(fi, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load or parse model: {e}")
# -----------------------------
# Predictions (works with or without DB)
# -----------------------------
st.divider()
st.subheader("🔮 Customer Churn Prediction")

left, right = st.columns([2, 1])

with left:
    if not df.empty and "user_id" in df.columns:
        user_id_options = sorted(filtered["user_id"].astype(str).dropna().unique())
        selected_user_id = st.selectbox("Choose a User ID", user_id_options, key="predict_user_select")
    else:
        selected_user_id = ""

    manual_user_id = st.text_input("Or type a User ID (optional)", "", key="predict_manual_user_id")
    user_to_predict = manual_user_id.strip() or selected_user_id

with right:
    st.write(" ")
    st.write(" ")
    do_predict = st.button("Predict Churn", key="predict_btn", use_container_width=True)

if do_predict:
    if not user_to_predict:
        st.warning("Please choose or type a user_id.")
    else:
        with st.spinner("Calling API (Render may take a moment to wake)..."):
            data = api_predict(user_to_predict)

        if not data or "error" in data:
            st.error(f"Prediction failed: {data.get('error', 'Unknown error')}")
            st.info("Tip: Render free tier sleeps. Try again after ~20–60 seconds.")
        else:
            prob = float(data.get("churn_probability", 0))
            risk = data.get("risk_band", "Unknown")
            threshold = float(data.get("threshold", 0.5))
            predicted = int(data.get("predicted_churn", 0))

            c1, c2, c3 = st.columns(3)
            c1.metric("Churn Probability", f"{prob:.2%}")
            c2.metric("Threshold Used", f"{threshold:.2f}")
            c3.metric("Predicted Churn", "YES" if predicted else "NO")

            if risk == "High":
                st.error(f"Risk Band: {risk}")
            elif risk == "Medium":
                st.warning(f"Risk Band: {risk}")
            else:
                st.info(f"Risk Band: {risk}")

# -----------------------------
# # -----------------------------
# High risk scorer (requires DB list of users)
# -----------------------------
st.divider()
st.subheader("🚨 Top High-Risk Users (from API)")

if df.empty or "user_id" not in df.columns:
    st.info("High-risk scoring needs the DB user list. Prediction above still works in API-only mode.")
else:
    top_n = st.slider("How many users to score?", 5, 50, 15, 5, key="score_topn")

    score_btn = st.button("Score users", key="score_btn")
    if score_btn:
        rows = []
        with st.spinner("Scoring users..."):
            for uid in filtered["user_id"].astype(str).head(top_n * 3):
                out = api_predict(uid)
                if out and "error" not in out:
                    rows.append(out)

        if rows:
            scored = pd.DataFrame(rows)
            scored["churn_probability"] = scored["churn_probability"].astype(float)
            scored = scored.sort_values("churn_probability", ascending=False).head(top_n)
            st.dataframe(scored, use_container_width=True)
        else:
            st.error("Could not score users. API may still be waking up. Try again in 30–60 seconds.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Developed by <b>Angel Michael</b> | "
    "<a href='https://github.com/Michaelossom' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)