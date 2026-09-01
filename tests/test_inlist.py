from pathlib import Path

import pandas as pd

from src.data.inlist import INLIST_NUM, add_inlist_features
from src.data.side import LEAK_COLS
from src.eval.metrics import ranking_metrics


def test_inlist_features_ignore_clicks_and_exclude_leaks():
    raw = Path("data/raw/KuaiRand-Pure/data")
    vocab = Path("data/processed/vocab.json")
    val = pd.read_parquet("data/processed/val.parquet").head(50).copy()
    flipped = val.copy()
    flipped["is_click"] = 1 - flipped["is_click"]
    a, cats, nums = add_inlist_features(val, raw, vocab)
    b, _, _ = add_inlist_features(flipped, raw, vocab)
    for col in ("tag_a", "server_width", "time_rank"):
        assert col in a.columns
        assert (a[col].fillna(-1).to_numpy() == b[col].fillna(-1).to_numpy()).all()
    assert not (LEAK_COLS & set(cats + nums))
    assert "time_rank" in INLIST_NUM


def test_ranking_primary_ndcg_with_eval_user_id():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0],
            "eval_user_id": [8, 8, 9, 9],
            "is_click": [1, 0, 1, 0],
            "pred": [0.7, 0.2, 0.9, 0.1],
        }
    )
    m = ranking_metrics(df)
    assert m["n_users"] == 2
    assert m["primary"] == m["ndcg@10"]
