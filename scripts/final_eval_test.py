"""ONE-SHOT hidden-test evaluation. Do not run during the agent search loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.prepare import ensure_prepared, load_config
from src.device import get_device
from src.eval.metrics import ranking_metrics
from src.models.deepfm import DeepFM
from src.models.gbdt import predict_gbdt
from src.models.popularity import predict_popularity
from src.train import _cat_matrix, load_splits, predict_deepfm


def _load_net(ckpt: dict, device):
    net = DeepFM(
        ckpt["field_dims"],
        embed_dim=int(ckpt.get("embed_dim", 16)),
        mlp_dims=tuple(ckpt.get("mlp_dims", (64, 32))),
        dropout=float(ckpt.get("dropout", 0.1)),
        n_heads=3 if ckpt.get("multitask") else 1,
    ).to(device)
    net.load_state_dict(ckpt["state_dict"])
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default=None, help="Override; default reads results/val_best.json")
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_prepared(args.config)
    splits, meta = load_splits(cfg, include_test=True)
    best_path = Path(cfg["paths"]["val_best_path"])
    model_name = args.model
    if model_name is None and best_path.exists():
        model_name = json.loads(best_path.read_text(encoding="utf-8"))["model"]
    model_name = model_name or "gbdt"

    df = splits["test"].copy()
    device = get_device()
    results_dir = Path(cfg["paths"]["results_dir"])
    if model_name == "popularity":
        blob = torch.load(results_dir / "popularity.pkl", weights_only=False)
        df["pred"] = predict_popularity(df, blob["item_score"], blob["global_ctr"])
    elif model_name == "gbdt":
        df["pred"] = predict_gbdt(df, results_dir / "gbdt.txt", meta["cat_cols"], meta.get("num_cols", []))
    else:
        ckpt_name = "multitask.pt" if model_name == "multitask" else "deepfm.pt"
        ckpt = torch.load(results_dir / ckpt_name, map_location=device, weights_only=False)
        net = _load_net(ckpt, device)
        x = torch.from_numpy(_cat_matrix(df, ckpt["cat_cols"]))
        df["pred"] = predict_deepfm(net, x, device, int(cfg["train"]["batch_size"]), bool(ckpt.get("multitask")))

    metrics = ranking_metrics(df, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])
    payload = {"model": model_name, **metrics}
    baseline_path = Path(cfg["paths"]["baseline_path"])
    if baseline_path.exists():
        payload["val_baseline"] = json.loads(baseline_path.read_text(encoding="utf-8"))
        payload["note"] = "val_baseline is the DeepFM click baseline on val, not test. Compare test numbers only against a test-time DeepFM run if you also score the baseline checkpoint here."
    out = results_dir / "test_once.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
