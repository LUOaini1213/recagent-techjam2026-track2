import pandas as pd

from src.data.history import add_history_features
from src.train import load_splits


def test_history_ctr_uses_train_labels_only():
    train = pd.DataFrame(
        {
            "eval_user_id": [1, 1, 2],
            "video_id": [10, 11, 10],
            "author_id": [5, 5, 6],
            "hour": [1, 1, 2],
            "is_click": [1, 0, 1],
        }
    )
    val = pd.DataFrame(
        {
            "eval_user_id": [1, 2],
            "video_id": [10, 10],
            "author_id": [5, 6],
            "hour": [1, 2],
            "is_click": [0, 0],
        }
    )
    tr, other, cols = add_history_features(train, val)
    assert "item_ctr" in cols
    train_v10 = float(tr.loc[tr["video_id"] == 10, "item_ctr"].iloc[0])
    val_v10 = float(other.loc[other["video_id"] == 10, "item_ctr"].iloc[0])
    assert val_v10 == train_v10
    # Val clicks are all 0; leaking them would pull CTR toward 0.5 from two extra negatives.
    assert val_v10 > 0.6


def test_load_splits_ranker_path_omits_test(tmp_path):
    import json

    processed = tmp_path / "p"
    processed.mkdir()
    for name, n in ("train", 4), ("val", 3), ("test", 80):
        pd.DataFrame(
            {
                "user_id": list(range(n)),
                "eval_user_id": list(range(n)),
                "video_id": list(range(n)),
                "is_click": [1, 0] * (n // 2) + ([1] if n % 2 else []),
            }
        ).to_parquet(processed / f"{name}.parquet", index=False)
    (processed / "meta.json").write_text(json.dumps({"cat_cols": [], "num_cols": []}), encoding="utf-8")
    splits, _ = load_splits({"paths": {"processed_dir": str(processed)}}, include_test=False)
    assert "test" not in splits
    assert set(splits) == {"train", "val"}
