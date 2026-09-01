from pathlib import Path

from src.data.prepare import prepare
from src.data.synthetic import write_synthetic_pure


def test_official_split_sizes(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    write_synthetic_pure(raw, n_users=10, n_items=12)
    # Point config via monkeypatch of yaml would be heavy; call featurize path through prepare
    # by writing a tiny config
    cfg = tmp_path / "cfg.yaml"
    processed = tmp_path / "proc"
    cfg.write_text(
        f"""
seed: 1
k_ndcg: 10
k_recall: 50
split:
  val_fraction: 0.5
paths:
  raw_dir: {raw.as_posix()}
  processed_dir: {processed.as_posix()}
""",
        encoding="utf-8",
    )
    meta = prepare(cfg, allow_synthetic=False)
    assert meta["n_train"] > 0
    assert meta["n_val"] > 0
    assert meta["n_test"] > 0
    assert meta["n_val"] + meta["n_test"] == 4000
    assert (processed / "train.parquet").exists()
    assert (processed / "vocab.json").exists()
    assert meta["val_impressions"]["n_users"] > 0
