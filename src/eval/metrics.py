"""Track 2 ranking metrics on logged impressions.

Official pair: click NDCG@10 and Recall@50, equal-weighted.
On KuaiRand-Pure val, most users have << 50 impressions (median ~5), so
Recall@50 saturates. Agent selection and early-stop use NDCG@10 as primary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ndcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    rel = np.asarray(relevances, dtype=np.float64)
    if rel.size == 0:
        return 0.0
    k = min(k, rel.size)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum((np.power(2.0, rel[:k]) - 1.0) / discounts))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum((np.power(2.0, ideal[:k]) - 1.0) / discounts))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(relevances: np.ndarray, k: int = 50) -> float:
    rel = np.asarray(relevances, dtype=np.float64)
    total = float((rel > 0).sum())
    if total == 0.0:
        return 0.0
    k = min(k, rel.size)
    return float((rel[:k] > 0).sum() / total)


def user_col(df: pd.DataFrame) -> str:
    return "eval_user_id" if "eval_user_id" in df.columns else "user_id"


def impression_profile(df: pd.DataFrame, user_col_name: str | None = None) -> dict:
    user_col_name = user_col_name or user_col(df)
    sizes = df.groupby(user_col_name, sort=False).size().to_numpy()
    return {
        "n_users": int(sizes.size),
        "n_rows": int(len(df)),
        "list_len_p50": float(np.median(sizes)),
        "list_len_p90": float(np.quantile(sizes, 0.90)),
        "list_len_max": int(sizes.max()) if sizes.size else 0,
        "frac_list_le_10": float((sizes <= 10).mean()) if sizes.size else 0.0,
        "frac_list_le_50": float((sizes <= 50).mean()) if sizes.size else 0.0,
    }


def ranking_metrics(
    df: pd.DataFrame,
    score_col: str = "pred",
    label_col: str = "is_click",
    k_ndcg: int = 10,
    k_recall: int = 50,
) -> dict:
    """Macro-average over users with at least one click.

    Also reports NDCG@5 (matches typical list length) and impression-list stats.
    `score` is the official 50/50 mix. `primary` is NDCG@10 for model selection.
    """
    uc = user_col(df)
    work = df[[uc, score_col, label_col]].to_numpy()
    order = np.lexsort((-work[:, 1].astype(np.float64), work[:, 0]))
    users = work[order, 0]
    scores = work[order, 1].astype(np.float64)
    labels = work[order, 2].astype(np.float64)
    bounds = np.flatnonzero(np.r_[True, users[1:] != users[:-1], True])

    ndcg10: list[float] = []
    ndcg5: list[float] = []
    recalls: list[float] = []
    list_lens: list[int] = []
    n_users = int(len(bounds) - 1)
    for i in range(n_users):
        sl = slice(int(bounds[i]), int(bounds[i + 1]))
        rel = labels[sl]
        list_lens.append(int(rel.size))
        if (rel > 0).sum() == 0:
            continue
        ndcg10.append(ndcg_at_k(rel, k_ndcg))
        ndcg5.append(ndcg_at_k(rel, 5))
        recalls.append(recall_at_k(rel, k_recall))

    profile = {
        "n_users": n_users,
        "n_rows": int(len(df)),
        "list_len_p50": float(np.median(list_lens)) if list_lens else 0.0,
        "list_len_p90": float(np.quantile(list_lens, 0.90)) if list_lens else 0.0,
        "frac_list_le_50": float((np.asarray(list_lens) <= 50).mean()) if list_lens else 0.0,
    }
    if not ndcg10:
        return {
            "ndcg@10": 0.0,
            "ndcg@5": 0.0,
            "recall@50": 0.0,
            "primary": 0.0,
            "score": 0.0,
            "n_users": n_users,
            "n_scored_users": 0,
            "impressions": profile,
        }

    ndcg = float(np.mean(ndcg10))
    recall = float(np.mean(recalls))
    auc = None
    pos = labels.sum()
    if 0.0 < pos < labels.size:
        # Wilcoxon-Mann-Whitney on the already-sorted-by-user list is wrong;
        # use global ranking by score.
        auc = _auc(df[label_col].to_numpy(np.float64), df[score_col].to_numpy(np.float64))
    return {
        "ndcg@10": ndcg,
        "ndcg@5": float(np.mean(ndcg5)),
        "recall@50": recall,
        "auc": auc,
        "primary": ndcg,
        "score": 0.5 * ndcg + 0.5 * recall,
        "n_users": n_users,
        "n_scored_users": len(ndcg10),
        "impressions": profile,
    }


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    ranks = np.arange(1, y.size + 1, dtype=np.float64)
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def delta_vs_baseline(metrics: dict, baseline: dict) -> dict:
    keys = ("ndcg@10", "ndcg@5", "recall@50", "primary", "score")
    return {k: float(metrics.get(k) or 0.0) - float(baseline.get(k) or 0.0) for k in keys}
