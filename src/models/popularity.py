from __future__ import annotations

import pandas as pd


def fit_popularity(train: pd.DataFrame) -> pd.Series:
    stats = train.groupby("video_id")["is_click"].agg(["sum", "count"])
    # Bayesian-smoothed item CTR
    global_ctr = float(train["is_click"].mean())
    alpha = 20.0
    score = (stats["sum"] + alpha * global_ctr) / (stats["count"] + alpha)
    return score


def predict_popularity(df: pd.DataFrame, item_score: pd.Series, global_ctr: float) -> pd.Series:
    return df["video_id"].map(item_score).fillna(global_ctr)
