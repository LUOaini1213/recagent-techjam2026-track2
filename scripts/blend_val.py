"""Sweep GBDT + item-CTR blend on validation only. Do not load test."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.prepare import ensure_prepared, load_config
from src.eval.metrics import ranking_metrics, user_col
from src.models.blend import WEIGHTS, attach_user_z, blend_scores
from src.models.gbdt import fit_predict_gbdt, predict_gbdt
from src.models.popularity import fit_popularity, predict_popularity
from src.train import load_splits


def main() -> None:
    import os

    os.chdir(ROOT)
    cfg_path = str(ROOT / "configs" / "default.yaml")
    cfg = load_config(cfg_path)
    meta = ensure_prepared(cfg_path)
    if meta.get("used_synthetic"):
        raise SystemExit("Refusing blend on synthetic data. Prepare KuaiRand-Pure first.")
    splits, meta = load_splits(cfg, include_test=False)
    results = Path(cfg["paths"]["results_dir"])
    results.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    item_score = fit_popularity(splits["train"])
    gctr = float(splits["train"]["is_click"].mean())
    val = splits["val"].copy()
    val["pop"] = predict_popularity(val, item_score, gctr).to_numpy(np.float64)

    gbdt_path = results / "gbdt.txt"
    if gbdt_path.exists():
        val["gbdt"] = predict_gbdt(val, gbdt_path, meta["cat_cols"], meta.get("num_cols", []))
    else:
        val["gbdt"] = fit_predict_gbdt(
            splits["train"],
            val,
            meta["cat_cols"],
            meta.get("num_cols", []),
            cfg["seed"],
            gbdt_path,
        )

    uc = user_col(val)
    val = attach_user_z(val, "gbdt", "pop", uc)

    rows = []
    best = None
    for w in WEIGHTS:
        tmp = val.copy()
        tmp["pred"] = blend_scores(tmp["gbdt_z"].to_numpy(), tmp["pop_z"].to_numpy(), w)
        m = ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])
        rec = {"w": w, "mode": "user_z", **{k: m[k] for k in ("ndcg@10", "ndcg@5", "recall@50", "primary", "score", "n_users", "n_scored_users")}}
        rows.append(rec)
        if best is None or rec["primary"] > best["primary"]:
            best = rec

        tmp["pred"] = blend_scores(tmp["gbdt"].to_numpy(np.float64), tmp["pop"].to_numpy(np.float64), w)
        m2 = ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])
        rec2 = {"w": w, "mode": "raw", **{k: m2[k] for k in ("ndcg@10", "ndcg@5", "recall@50", "primary", "score", "n_users", "n_scored_users")}}
        rows.append(rec2)
        if rec2["primary"] > best["primary"]:
            best = rec2

    # Pure ends of the grid for the write-up
    for name, col in ("gbdt_only", "gbdt"), ("pop_only", "pop"):
        tmp = val.copy()
        tmp["pred"] = tmp[col]
        m = ranking_metrics(tmp, k_ndcg=cfg["k_ndcg"], k_recall=cfg["k_recall"])
        rec = {"w": 1.0 if name == "gbdt_only" else 0.0, "mode": name, **{k: m[k] for k in ("ndcg@10", "ndcg@5", "recall@50", "primary", "score", "n_users", "n_scored_users")}}
        rows.append(rec)
        if rec["primary"] > best["primary"]:
            best = rec

    payload = {
        "best": best,
        "grid": rows,
        "wall_seconds": round(time.time() - t0, 3),
        "n_val": int(len(val)),
        "used_synthetic": False,
        "note": "Val only. Test labels were not loaded.",
    }
    (results / "blend_val.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    model_name = "gbdt" if best and (best.get("mode") in ("gbdt_only", "user_z", "raw") and float(best.get("w", 1)) >= 0.999) else "blend"
    if best and best.get("mode") == "pop_only":
        model_name = "popularity"
    (results / "val_best.json").write_text(json.dumps({"model": model_name, **best}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
