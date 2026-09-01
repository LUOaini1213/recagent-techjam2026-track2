from pathlib import Path

import pandas as pd

from src.data.side import LEAK_COLS, add_side_features, side_feature_names
from src.eval.metrics import ranking_metrics


def test_side_join_ignores_val_click_and_avoids_leak_names(tmp_path: Path):
    raw = Path("data/raw/KuaiRand-Pure/data")
    vocab = Path("data/processed/vocab.json")
    if not raw.exists() or not vocab.exists():
        return
    val = pd.read_parquet("data/processed/val.parquet").head(40).copy()
    flipped = val.copy()
    flipped["is_click"] = 1 - flipped["is_click"]
    a, cats, nums = add_side_features(val, raw, vocab)
    b, _, _ = add_side_features(flipped, raw, vocab)
    for col in ("onehot_feat0", "register_days", "share_cnt"):
        if col in a.columns:
            assert (a[col].fillna(-1).to_numpy() == b[col].fillna(-1).to_numpy()).all()
    assert not (LEAK_COLS & set(cats + nums))
    assert "onehot_feat0" in side_feature_names()


def test_ranking_primary_still_ndcg_on_short_lists():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0],
            "eval_user_id": [3, 3, 4, 4],
            "is_click": [1, 0, 1, 0],
            "pred": [0.9, 0.1, 0.8, 0.2],
        }
    )
    m = ranking_metrics(df)
    assert m["n_users"] == 2
    assert m["primary"] == m["ndcg@10"]
