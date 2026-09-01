"""Train-only user/item history features. Val/test must not contribute labels or counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_COLS = [
    "item_ctr",
    "item_n",
    "user_ctr",
    "user_n",
    "user_author_ctr",
    "user_hour_ctr",
    "time_rank",
    "time_z",
]


def add_history_features(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fit CTR / count maps on `train` only, then attach to train and `other` (val)."""
    user_col = "eval_user_id" if "eval_user_id" in train.columns else "user_id"
    gctr = float(train["is_click"].mean())
    alpha = 20.0

    item = train.groupby("video_id", sort=False)["is_click"].agg(["sum", "count"])
    item["item_ctr"] = (item["sum"] + alpha * gctr) / (item["count"] + alpha)
    item["item_n"] = np.log1p(item["count"])

    user = train.groupby(user_col, sort=False)["is_click"].agg(["sum", "count"])
    user["user_ctr"] = (user["sum"] + alpha * gctr) / (user["count"] + alpha)
    user["user_n"] = np.log1p(user["count"])

    ua = None
    if "author_id" in train.columns:
        tmp = train.groupby([user_col, "author_id"], sort=False)["is_click"].agg(["sum", "count"]).reset_index()
        tmp["user_author_ctr"] = (tmp["sum"] + alpha * gctr) / (tmp["count"] + alpha)
        ua = tmp[[user_col, "author_id", "user_author_ctr"]]

    uh = None
    if "hour" in train.columns:
        tmp = train.groupby([user_col, "hour"], sort=False)["is_click"].agg(["sum", "count"]).reset_index()
        tmp["user_hour_ctr"] = (tmp["sum"] + alpha * gctr) / (tmp["count"] + alpha)
        uh = tmp[[user_col, "hour", "user_hour_ctr"]]

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["item_ctr"] = out["video_id"].map(item["item_ctr"]).fillna(gctr)
        out["item_n"] = out["video_id"].map(item["item_n"]).fillna(0.0)
        out["user_ctr"] = out[user_col].map(user["user_ctr"]).fillna(gctr)
        out["user_n"] = out[user_col].map(user["user_n"]).fillna(0.0)
        if ua is not None:
            out = out.merge(ua, on=[user_col, "author_id"], how="left")
            out["user_author_ctr"] = out["user_author_ctr"].fillna(gctr)
        else:
            out["user_author_ctr"] = gctr
        if uh is not None:
            out = out.merge(uh, on=[user_col, "hour"], how="left")
            out["user_hour_ctr"] = out["user_hour_ctr"].fillna(gctr)
        else:
            out["user_hour_ctr"] = gctr
        if "time_ms" in out.columns:
            uc = user_col
            out["time_rank"] = out.groupby(uc, sort=False)["time_ms"].rank(method="first")
            mu = out.groupby(uc, sort=False)["time_ms"].transform("mean")
            sd = out.groupby(uc, sort=False)["time_ms"].transform("std").replace(0, 1.0).fillna(1.0)
            out["time_z"] = (out["time_ms"] - mu) / sd
        else:
            out["time_rank"] = 0.0
            out["time_z"] = 0.0
        return out

    return apply(train), apply(other), list(HISTORY_COLS)
