from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.prepare import ensure_prepared, load_config
from src.device import cuda_status, get_device
from src.eval.metrics import ranking_metrics
from src.models.deepfm import DeepFM
from src.models.gbdt import (
    fit_predict_gbdt,
    fit_predict_gbdt_hist,
    fit_predict_gbdt_inlist,
    fit_predict_gbdt_side,
    fit_predict_ranker,
    predict_gbdt,
)
from src.models.popularity import fit_popularity, predict_popularity
from src.models.rerank import search_rerank

AUX_WEIGHT = {"is_like": 0.2, "long_view": 0.2}


def load_splits(cfg: dict, include_test: bool = False):
    d = Path(cfg["paths"]["processed_dir"])
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    names = ("train", "val", "test") if include_test else ("train", "val")
    splits = {k: pd.read_parquet(d / f"{k}.parquet") for k in names}
    return splits, meta


def _cat_matrix(df: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    return df[cat_cols].to_numpy(dtype=np.int64)


def _cpu_limits(cfg: dict, device: torch.device) -> tuple[int, int | None]:
    tcfg = cfg["train"]
    epochs = int(tcfg["max_epochs"])
    subsample = None
    if device.type == "cpu":
        epochs = int(tcfg.get("cpu_max_epochs", epochs))
        n = int(tcfg.get("cpu_subsample_rows", 0) or 0)
        subsample = n if n > 0 else None
    return epochs, subsample


def train_popularity(splits: dict, cfg: dict) -> dict:
    item_score = fit_popularity(splits["train"])
    g = float(splits["train"]["is_click"].mean())
    tmp = splits["val"].copy()
    tmp["pred"] = predict_popularity(tmp, item_score, g).to_numpy()
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    torch.save({"item_score": item_score, "global_ctr": g, "model": "popularity"}, Path(cfg["paths"]["results_dir"]) / "popularity.pkl")
    return ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])


def train_gbdt(splits: dict, meta: dict, cfg: dict) -> dict:
    pred = fit_predict_gbdt(
        splits["train"],
        splits["val"],
        meta["cat_cols"],
        meta.get("num_cols", []),
        cfg["seed"],
        Path(cfg["paths"]["results_dir"]) / "gbdt.txt",
    )
    tmp = splits["val"].copy()
    tmp["pred"] = pred
    return ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])


def train_gbdt_side(splits: dict, meta: dict, cfg: dict) -> dict:
    pred = fit_predict_gbdt_side(
        splits["train"],
        splits["val"],
        meta["cat_cols"],
        meta.get("num_cols", []),
        cfg["seed"],
        Path(cfg["paths"]["results_dir"]) / "gbdt_side.txt",
        Path(cfg["paths"]["raw_dir"]),
        Path(cfg["paths"]["processed_dir"]) / "vocab.json",
    )
    tmp = splits["val"].copy()
    tmp["pred"] = pred
    return ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])


def train_gbdt_inlist(splits: dict, meta: dict, cfg: dict, mode: str = "res") -> dict:
    pred = fit_predict_gbdt_inlist(
        splits["train"],
        splits["val"],
        meta["cat_cols"],
        meta.get("num_cols", []),
        cfg["seed"],
        Path(cfg["paths"]["results_dir"]) / f"gbdt_inlist_{mode}.txt",
        Path(cfg["paths"]["raw_dir"]),
        Path(cfg["paths"]["processed_dir"]) / "vocab.json",
        mode=mode,
    )
    tmp = splits["val"].copy()
    tmp["pred"] = pred
    return ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])


def train_gbdt_hist(splits: dict, meta: dict, cfg: dict) -> dict:
    pred = fit_predict_gbdt_hist(
        splits["train"],
        splits["val"],
        meta["cat_cols"],
        meta.get("num_cols", []),
        cfg["seed"],
        Path(cfg["paths"]["results_dir"]) / "gbdt_hist.txt",
    )
    tmp = splits["val"].copy()
    tmp["pred"] = pred
    return ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])


def train_ranker(splits: dict, meta: dict, cfg: dict) -> dict:
    pred = fit_predict_ranker(
        splits["train"],
        splits["val"],
        meta["cat_cols"],
        meta.get("num_cols", []),
        cfg["seed"],
        Path(cfg["paths"]["results_dir"]) / "ranker.txt",
    )
    tmp = splits["val"].copy()
    tmp["pred"] = pred
    return ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])


def train_gbdt_rerank(splits: dict, meta: dict, cfg: dict) -> dict:
    """Frozen pointwise GBDT + val-only time/CTR residual. Does not read test."""
    results = Path(cfg["paths"]["results_dir"])
    gbdt_path = results / "gbdt.txt"
    if not gbdt_path.exists():
        fit_predict_gbdt(
            splits["train"],
            splits["val"],
            meta["cat_cols"],
            meta.get("num_cols", []),
            cfg["seed"],
            gbdt_path,
        )
    val = splits["val"].copy()
    val["gbdt"] = predict_gbdt(val, gbdt_path, meta["cat_cols"], meta.get("num_cols", []))
    item = fit_popularity(splits["train"])
    gctr = float(splits["train"]["is_click"].mean())
    val["pop"] = predict_popularity(val, item, gctr)
    uc = "eval_user_id" if "eval_user_id" in val.columns else "user_id"
    val["time_rank"] = val.groupby(uc, sort=False)["time_ms"].rank(method="first") if "time_ms" in val.columns else 0.0
    best = search_rerank(val, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])
    best["model"] = "gbdt_rerank"
    (results / "gbdt_rerank_val.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def train_deepfm(
    splits: dict,
    meta: dict,
    cfg: dict,
    multitask: bool = False,
    batch_size: int | None = None,
) -> dict:
    device = get_device()
    cat_cols = meta["cat_cols"]
    field_dims = [meta["field_dims"][c] for c in cat_cols]
    tcfg = cfg["train"]
    batch_size = batch_size or int(tcfg["batch_size"])
    epochs, subsample = _cpu_limits(cfg, device)
    train_df = splits["train"]
    if subsample and len(train_df) > subsample:
        train_df = train_df.sample(n=subsample, random_state=cfg["seed"])

    n_heads = 3 if multitask else 1
    embed_dim = int(tcfg["embed_dim"])
    mlp_dims = tuple(tcfg["mlp_dims"])
    model = DeepFM(
        field_dims,
        embed_dim=embed_dim,
        mlp_dims=mlp_dims,
        dropout=float(tcfg["dropout"]),
        n_heads=n_heads,
    ).to(device)

    x = torch.from_numpy(_cat_matrix(train_df, cat_cols))
    y_click = torch.from_numpy(train_df["is_click"].to_numpy(np.float32))
    y_like = torch.from_numpy(train_df["is_like"].to_numpy(np.float32))
    y_lv = torch.from_numpy(train_df["long_view"].to_numpy(np.float32))
    loader = DataLoader(TensorDataset(x, y_click, y_like, y_lv), batch_size=batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    bce = nn.BCEWithLogitsLoss()
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    x_val = torch.from_numpy(_cat_matrix(splits["val"], cat_cols))

    best = {"primary": -1.0}
    stale = 0
    ckpt = Path(cfg["paths"]["results_dir"]) / ("multitask.pt" if multitask else "deepfm.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for xb, yc, yl, yv in loader:
            xb, yc, yl, yv = xb.to(device), yc.to(device), yl.to(device), yv.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                if multitask:
                    loss = (
                        bce(logits[:, 0], yc)
                        + AUX_WEIGHT["is_like"] * bce(logits[:, 1], yl)
                        + AUX_WEIGHT["long_view"] * bce(logits[:, 2], yv)
                    )
                else:
                    loss = bce(logits, yc)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        preds = predict_deepfm(model, x_val, device, batch_size, multitask)
        tmp = splits["val"].copy()
        tmp["pred"] = preds
        metrics = ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])
        if metrics["primary"] > best["primary"] + 1e-12:
            best = {**metrics, "epoch": epoch}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "cat_cols": cat_cols,
                    "field_dims": field_dims,
                    "multitask": multitask,
                    "embed_dim": embed_dim,
                    "mlp_dims": list(mlp_dims),
                    "dropout": float(tcfg["dropout"]),
                },
                ckpt,
            )
            stale = 0
        else:
            stale += 1
            if stale >= int(tcfg["early_stop"]):
                break
    keep = ("ndcg@10", "ndcg@5", "recall@50", "auc", "primary", "score", "n_users", "n_scored_users", "impressions", "epoch")
    return {k: best[k] for k in keep if k in best}


@torch.no_grad()
def predict_deepfm(model, x_val, device, batch_size, multitask: bool) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, x_val.size(0), batch_size):
        xb = x_val[i : i + batch_size].to(device)
        logits = model(xb)
        if multitask:
            logits = logits[:, 0]
        outs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def run(model_name: str, config_path: str = "configs/default.yaml", batch_size: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_prepared(config_path)
    splits, meta = load_splits(cfg, include_test=False)
    t0 = time.time()
    if model_name == "popularity":
        metrics = train_popularity(splits, cfg)
    elif model_name == "gbdt":
        metrics = train_gbdt(splits, meta, cfg)
    elif model_name == "gbdt_hist":
        metrics = train_gbdt_hist(splits, meta, cfg)
    elif model_name == "gbdt_side":
        metrics = train_gbdt_side(splits, meta, cfg)
    elif model_name.startswith("gbdt_inlist"):
        mode = "all"
        if model_name in ("gbdt_inlist_tags", "gbdt_inlist_res", "gbdt_inlist_time"):
            mode = model_name.rsplit("_", 1)[-1]
        elif model_name == "gbdt_inlist":
            mode = "res"
        metrics = train_gbdt_inlist(splits, meta, cfg, mode=mode)
        metrics["model"] = model_name
    elif model_name == "ranker":
        metrics = train_ranker(splits, meta, cfg)
    elif model_name == "gbdt_rerank":
        metrics = train_gbdt_rerank(splits, meta, cfg)
    elif model_name == "multitask":
        metrics = train_deepfm(splits, meta, cfg, multitask=True, batch_size=batch_size)
    else:
        metrics = train_deepfm(splits, meta, cfg, multitask=False, batch_size=batch_size)
    metrics["wall_seconds"] = round(time.time() - t0, 3)
    metrics["device"] = cuda_status()
    metrics["model"] = model_name
    out_path = Path(cfg["paths"]["results_dir"]) / f"{model_name}_val.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="gbdt",
        choices=[
            "popularity",
            "deepfm",
            "multitask",
            "gbdt",
            "gbdt_hist",
            "gbdt_side",
            "gbdt_inlist",
            "gbdt_inlist_tags",
            "gbdt_inlist_res",
            "gbdt_inlist_time",
            "ranker",
            "gbdt_rerank",
        ],
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--force-prepare", action="store_true")
    args = p.parse_args()
    if args.force_prepare:
        from src.data.prepare import prepare

        prepare(args.config)
    print(json.dumps(run(args.model, args.config, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
