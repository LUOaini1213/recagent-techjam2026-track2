"""Train-only categorical vocab. Index 0 is always OOV / unseen."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def fit_vocab(train: pd.DataFrame, columns: Iterable[str]) -> dict[str, dict[str, int]]:
    vocabs: dict[str, dict[str, int]] = {}
    for col in columns:
        if col not in train.columns:
            continue
        values = pd.unique(train[col].astype(str).to_numpy())
        vocabs[col] = {str(v): i + 1 for i, v in enumerate(values)}
    return vocabs


def transform_col(series: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    mapped = series.astype(str).map(mapping)
    return mapped.fillna(0).astype(np.int64).to_numpy()


def field_dims(vocabs: dict[str, dict[str, int]]) -> dict[str, int]:
    # +1 for the OOV bucket at index 0
    return {col: int(len(mp) + 1) for col, mp in vocabs.items()}
