import numpy as np
import pandas as pd


def _safe_div(a, b):
    b = np.where(b == 0, 1, b)
    return a / b


def build_user_features(
    riders: pd.DataFrame, trips: pd.DataFrame, sessions: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds one row per user_id:
    - RFM-like trip features
    - Weekday vs weekend behavior
    - Session engagement features
    - Tenure + basic rider attributes
    Also creates churn_30d label: 30+ days since last trip (inactivity).
    """

    # --- Basic checks ---
    if "user_id" not in riders.columns:
        raise ValueError("riders.csv must contain 'user_id' column")
    if "user_id" not in trips.columns:
        raise ValueError("trips.csv must contain 'user_id' column")
    if "pickup_time" not in trips.columns:
        raise ValueError("trips.csv must contain 'pickup_time' column")
    if "rider_id" not in sessions.columns:
        raise ValueError("sessions.csv must contain 'rider_id' column")
    if "session_time" not in sessions.columns:
        raise ValueError("sessions.csv must contain 'session_time' column")

    # --- Snapshot date (latest trip time) ---
    # Convert pickup_time to UTC (tz-aware), then drop timezone to make it tz-naive
    t_pickup = pd.to_datetime(trips["pickup_time"], errors="coerce", utc=True)
    snapshot = t_pickup.max().tz_localize(None)

    # ---------- TRIP FEATURES ----------
    t = trips.copy()
    t["pickup_time"] = (
        pd.to_datetime(t["pickup_time"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    t = t.dropna(subset=["user_id", "pickup_time"])

    # Some datasets use different names; handle gracefully
    trip_id_col = "trip_id" if "trip_id" in t.columns else None
    fare_col = "fare" if "fare" in t.columns else None
    surge_col = "surge_multiplier" if "surge_multiplier" in t.columns else None
    tip_col = "tip" if "tip" in t.columns else None

    t["dow"] = t["pickup_time"].dt.dayofweek
    t["is_weekend"] = (t["dow"] >= 5).astype(int)

    agg_dict = {
        "last_trip_time": ("pickup_time", "max"),
        "first_trip_time": ("pickup_time", "min"),
        "weekend_trips": ("is_weekend", "sum"),
    }

    # trips_count
    if trip_id_col:
        agg_dict["trips_count"] = (trip_id_col, "count")
    else:
        agg_dict["trips_count"] = ("pickup_time", "count")

    # fare/surge/tip
    if fare_col:
        agg_dict["total_fare"] = (fare_col, "sum")
        agg_dict["avg_fare"] = (fare_col, "mean")
    else:
        agg_dict["total_fare"] = ("pickup_time", "count")  # placeholder
        agg_dict["avg_fare"] = ("pickup_time", "count")

    if surge_col:
        agg_dict["avg_surge"] = (surge_col, "mean")
    else:
        agg_dict["avg_surge"] = ("pickup_time", "count")

    if tip_col:
        agg_dict["total_tip"] = (tip_col, "sum")
    else:
        agg_dict["total_tip"] = ("pickup_time", "count")

    trip_agg = t.groupby("user_id").agg(**agg_dict).reset_index()

    # Fix placeholders if fare/surge/tip missing
    if not fare_col:
        trip_agg["total_fare"] = 0.0
        trip_agg["avg_fare"] = 0.0
    if not surge_col:
        trip_agg["avg_surge"] = 0.0
    if not tip_col:
        trip_agg["total_tip"] = 0.0

    trip_agg["recency_days"] = (snapshot - trip_agg["last_trip_time"]).dt.days.fillna(0).astype(int)
    trip_agg["active_span_days"] = (trip_agg["last_trip_time"] - trip_agg["first_trip_time"]).dt.days.fillna(0).astype(int)
    trip_agg["weekend_share"] = _safe_div(
        trip_agg["weekend_trips"].astype(float),
        trip_agg["trips_count"].astype(float),
    )

    # Churn label: 30+ days inactivity
    trip_agg["churn_30d"] = (trip_agg["recency_days"] >= 30).astype(int)

    # ---------- SESSION FEATURES ----------
    s = sessions.copy()
    s["session_time"] = (
        pd.to_datetime(s["session_time"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    s = s.dropna(subset=["rider_id", "session_time"]).rename(columns={"rider_id": "user_id"})

    if "converted" in s.columns:
        s["converted"] = pd.to_numeric(s["converted"], errors="coerce").fillna(0).astype(int)
    else:
        s["converted"] = 0

    # Optional columns
    time_on_app_col = "time_on_app" if "time_on_app" in s.columns else None
    pages_col = "pages_visited" if "pages_visited" in s.columns else None

    sess_agg = s.groupby("user_id").agg(
        sessions_count=("session_time", "count"),
        last_session_time=("session_time", "max"),
        conversion_rate=("converted", "mean"),
        avg_time_on_app=(time_on_app_col, "mean") if time_on_app_col else ("converted", "mean"),
        avg_pages_visited=(pages_col, "mean") if pages_col else ("converted", "mean"),
    ).reset_index()

    if not time_on_app_col:
        sess_agg["avg_time_on_app"] = 0.0
    if not pages_col:
        sess_agg["avg_pages_visited"] = 0.0

    sess_agg["session_recency_days"] = (snapshot - sess_agg["last_session_time"]).dt.days.fillna(0).astype(int)

    # ---------- RIDER BASE ----------
    base = riders.copy()

    keep_cols = ["user_id", "signup_date", "loyalty_status", "age", "city", "avg_rating_given", "referred_by"]
    keep_cols = [c for c in keep_cols if c in base.columns]
    base = base[keep_cols].copy()

    if "signup_date" in base.columns:
        base["tenure_days"] = (
            snapshot - pd.to_datetime(base["signup_date"], errors="coerce", utc=True).dt.tz_localize(None)
        ).dt.days.fillna(0).astype(int)
    else:
        base["tenure_days"] = 0

    # Merge everything
    df = base.merge(trip_agg, on="user_id", how="left").merge(sess_agg, on="user_id", how="left")

    # Fill missing users with no activity
    fill_zero = [
        "trips_count", "total_fare", "avg_fare", "avg_surge", "total_tip",
        "weekend_trips", "weekend_share", "recency_days", "active_span_days", "churn_30d",
        "sessions_count", "avg_time_on_app", "avg_pages_visited", "conversion_rate", "session_recency_days",
    ]
    for c in fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Extra ratios
    df["spend_per_trip"] = np.where(df["trips_count"] == 0, 0, df["total_fare"] / df["trips_count"])
    df["tips_per_trip"] = np.where(df["trips_count"] == 0, 0, df["total_tip"] / df["trips_count"])

    return df
