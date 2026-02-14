import numpy as np
import pandas as pd


def assign_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segments aligned with your slides:
    - Regular Commuters: frequent trips, mostly weekdays
    - Weekend Users: high weekend share
    - Occasional Users: everything else (often higher churn risk)
    """

    out = df.copy()

    # Safety defaults
    if "trips_count" not in out.columns:
        out["trips_count"] = 0
    if "weekend_share" not in out.columns:
        out["weekend_share"] = 0.0

    freq = out["trips_count"].fillna(0)
    weekend_share = out["weekend_share"].fillna(0.0)

    # Heuristics (you can tune later)
    regular = (freq >= 8) & (weekend_share < 0.35)
    weekend = (weekend_share >= 0.55) & (freq >= 3)
    occasional = ~regular & ~weekend

    out["segment"] = np.select(
        [regular, weekend, occasional],
        ["Regular Commuters", "Weekend Users", "Occasional Users"],
        default="Occasional Users"
    )

    return out
