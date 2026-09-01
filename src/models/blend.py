"""Val-only score fusion. Never reads test labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _zscore_by_user(df: pd.DataFrame, col: str, user_col: str) -> np.ndarray:
    g = df.groupby(user_col, sort=False)[col]
    mean = g.transform("mean").to_numpy(np.float64)
    std = g.transform("std").to_numpy(np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    return (df[col].to_numpy(np.float64) - mean) / std


def blend_scores(gbdt: np.ndarray, pop: np.ndarray, w: float) -> np.ndarray:
    return w * gbdt + (1.0 - w) * pop


def attach_user_z(df: pd.DataFrame, gbdt_col: str, pop_col: str, user_col: str) -> pd.DataFrame:
    out = df.copy()
    out["gbdt_z"] = _zscore_by_user(out, gbdt_col, user_col)
    out["pop_z"] = _zscore_by_user(out, pop_col, user_col)
    return out
