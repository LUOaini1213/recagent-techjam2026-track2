"""Val-only linear rerank of a frozen GBDT score. Coefficients are not fit on test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import ranking_metrics, user_col

TIME_COEFS = np.linspace(-0.5, 0.5, 11)
POP_COEFS = np.linspace(-0.2, 0.2, 9)


def user_z(df: pd.DataFrame, col: str, user_col_name: str) -> np.ndarray:
    mu = df.groupby(user_col_name, sort=False)[col].transform("mean").to_numpy(np.float64)
    sd = df.groupby(user_col_name, sort=False)[col].transform("std").to_numpy(np.float64)
    sd = np.where(~np.isfinite(sd) | (sd < 1e-8), 1.0, sd)
    return (df[col].to_numpy(np.float64) - mu) / sd


def search_rerank(val: pd.DataFrame, k_ndcg: int = 10, k_recall: int = 50) -> dict:
    uc = user_col(val)
    work = val.copy()
    work["gbdt_z"] = user_z(work, "gbdt", uc)
    work["pop_z"] = user_z(work, "pop", uc)
    if "time_rank" not in work.columns:
        work["time_rank"] = work.groupby(uc, sort=False)["time_ms"].rank(method="first") if "time_ms" in work.columns else 0.0
    work["time_z"] = user_z(work, "time_rank", uc)

    best = None
    for a in TIME_COEFS:
        for b in POP_COEFS:
            work["pred"] = work["gbdt_z"] + float(a) * work["time_z"] + float(b) * work["pop_z"]
            m = ranking_metrics(work, k_ndcg=k_ndcg, k_recall=k_recall)
            rec = {
                "time_coef": float(a),
                "pop_coef": float(b),
                "ndcg@10": m["ndcg@10"],
                "ndcg@5": m["ndcg@5"],
                "recall@50": m["recall@50"],
                "primary": m["primary"],
                "score": m["score"],
                "n_users": m["n_users"],
                "n_scored_users": m["n_scored_users"],
            }
            if best is None or rec["primary"] > best["primary"]:
                best = rec
    return best
