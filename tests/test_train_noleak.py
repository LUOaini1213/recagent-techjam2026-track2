"""Shipped trainer must not pull hidden-test rows into the training loop."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.train import load_splits, train_popularity


def test_load_splits_default_omits_test(tmp_path: Path):
    processed = tmp_path / "processed"
    processed.mkdir()
    for name, n in [("train", 8), ("val", 6), ("test", 99)]:
        pd.DataFrame(
            {
                "user_id": list(range(n)),
                "eval_user_id": list(range(n)),
                "video_id": list(range(n)),
                "is_click": [1, 0] * (n // 2) + ([0] if n % 2 else []),
            }
        ).to_parquet(processed / f"{name}.parquet", index=False)
    (processed / "meta.json").write_text(json.dumps({"cat_cols": ["user_id"], "num_cols": []}), encoding="utf-8")
    cfg = {
        "paths": {"processed_dir": str(processed), "results_dir": str(tmp_path / "results")},
        "k_ndcg": 10,
        "k_recall": 50,
        "seed": 1,
    }
    splits, _meta = load_splits(cfg, include_test=False)
    assert set(splits) == {"train", "val"}
    assert "test" not in splits
    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 6


def test_popularity_uses_click_and_primary_is_ndcg(tmp_path: Path):
    results = tmp_path / "results"
    splits = {
        "train": pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "eval_user_id": [1, 1, 2, 2],
                "video_id": [10, 11, 10, 12],
                "is_click": [1, 0, 1, 0],
            }
        ),
        "val": pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "eval_user_id": [1, 1, 2, 2],
                "video_id": [10, 11, 10, 12],
                "is_click": [1, 0, 1, 0],
                "pred": [0.0, 0.0, 0.0, 0.0],
            }
        ),
    }
    cfg = {"paths": {"results_dir": str(results)}, "k_ndcg": 10, "k_recall": 50}
    metrics = train_popularity(splits, cfg)
    assert metrics["primary"] == metrics["ndcg@10"]
    assert "recall@50" in metrics
    assert metrics["n_users"] == 2
