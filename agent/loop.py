"""Autonomous experiment loop: inspect -> baseline -> explore -> log."""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path

from agent.catalog import CATALOG
from agent.logger import append_run
from src.data.prepare import ensure_prepared, load_config
from src.device import cuda_status
from src.eval.metrics import delta_vs_baseline
from src.train import run as run_train

OOM_RE = re.compile(r"out of memory", re.I)


def _primary(metrics: dict | None) -> float:
    if not metrics:
        return float("-inf")
    if metrics.get("primary") is not None:
        return float(metrics["primary"])
    return float(metrics.get("ndcg@10") or 0.0)


def _metric_slice(metrics: dict) -> dict:
    keys = (
        "ndcg@10",
        "ndcg@5",
        "recall@50",
        "auc",
        "primary",
        "score",
        "n_scored_users",
        "wall_seconds",
        "impressions",
        "epoch",
    )
    return {k: metrics.get(k) for k in keys if k in metrics}


def _fresh_outputs(cfg: dict) -> None:
    """Drop stale synthetic-run artifacts so a bot run is comparable on full val."""
    Path("logs").mkdir(exist_ok=True)
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    for key in ("log_path", "baseline_path", "val_best_path"):
        p = Path(cfg["paths"][key])
        if p.exists():
            p.unlink()
    extra = Path(cfg["paths"]["results_dir"]) / "agent_summary.json"
    if extra.exists():
        extra.unlink()


def run_agent(
    config_path: str = "configs/default.yaml",
    max_iters: int | None = None,
    fresh: bool = False,
) -> dict:
    cfg = load_config(config_path)
    meta = ensure_prepared(config_path)
    if fresh:
        _fresh_outputs(cfg)
    log_path = cfg["paths"]["log_path"]
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    cuda = bool(cuda_status()["cuda_available"])
    baseline = None
    best = None
    stale = 0
    history = []
    t_start = time.time()
    items = CATALOG if max_iters is None else CATALOG[:max_iters]

    for i, spec in enumerate(items):
        if spec["model"] is None:
            record = {
                "iter": i,
                "id": spec["id"],
                "role": spec["role"],
                "hypothesis": spec["hypothesis"],
                "why": spec["why"],
                "model": None,
                "metrics": None,
                "data": {
                    "n_train": meta.get("n_train"),
                    "n_val": meta.get("n_val"),
                    "click_rate_train": meta.get("click_rate_train"),
                    "used_synthetic": meta.get("used_synthetic"),
                    "val_impressions": meta.get("val_impressions"),
                },
                "error": None,
                "recovery": None,
                "human_interventions": 0,
                "tokens": {"in": 0, "out": 0},
                "gpu_hours": 0.0,
                "device": cuda_status(),
            }
            append_run(log_path, record)
            history.append(record)
            continue

        if spec.get("requires_gpu") and not cuda:
            record = {
                "iter": i,
                "id": spec["id"],
                "role": spec["role"],
                "hypothesis": spec["hypothesis"],
                "why": spec["why"],
                "model": spec["model"],
                "metrics": None,
                "error": None,
                "recovery": "skipped: requires_gpu and CUDA is unavailable",
                "human_interventions": 0,
                "tokens": {"in": 0, "out": 0},
                "gpu_hours": 0.0,
                "device": cuda_status(),
            }
            append_run(log_path, record)
            history.append(record)
            continue

        batch_size = int(cfg["train"]["batch_size"])
        error = None
        recovery = None
        metrics = None
        for _attempt in range(3):
            try:
                metrics = run_train(spec["model"], config_path, batch_size=batch_size)
                error = None
                break
            except RuntimeError as exc:
                error = str(exc)
                if OOM_RE.search(error) or "CUDA" in error:
                    batch_size = max(32, batch_size // 2)
                    recovery = f"OOM/CUDA -> retry batch_size={batch_size}"
                    continue
                recovery = "non-OOM failure, skip hypothesis"
                traceback.print_exc()
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                recovery = "exception, skip hypothesis"
                traceback.print_exc()
                break

        if metrics is None:
            record = {
                "iter": i,
                "id": spec["id"],
                "role": spec["role"],
                "hypothesis": spec["hypothesis"],
                "why": spec["why"],
                "model": spec["model"],
                "metrics": None,
                "error": error,
                "recovery": recovery,
                "human_interventions": 0,
                "tokens": {"in": 0, "out": 0},
                "gpu_hours": 0.0,
                "device": cuda_status(),
            }
            append_run(log_path, record)
            history.append(record)
            continue

        if spec["role"] == "baseline":
            baseline = {k: metrics.get(k) for k in ("ndcg@10", "ndcg@5", "recall@50", "primary", "score")}
            Path(cfg["paths"]["baseline_path"]).write_text(json.dumps(baseline, indent=2), encoding="utf-8")

        delta = delta_vs_baseline(metrics, baseline or {})
        gpu_hours = (metrics.get("wall_seconds") or 0) / 3600.0 if cuda else 0.0
        record = {
            "iter": i,
            "id": spec["id"],
            "role": spec["role"],
            "hypothesis": spec["hypothesis"],
            "why": spec["why"],
            "model": spec["model"],
            "metrics": _metric_slice(metrics),
            "delta_vs_baseline": delta,
            "error": error,
            "recovery": recovery,
            "human_interventions": 0,
            "tokens": {"in": 0, "out": 0},
            "gpu_hours": round(gpu_hours, 6),
            "cpu_wall_seconds": metrics.get("wall_seconds"),
            "device": metrics.get("device"),
        }
        append_run(log_path, record)
        history.append(record)

        if spec["role"] == "probe":
            continue
        if best is None or _primary(metrics) > _primary(best.get("metrics")) + cfg["convergence"]["epsilon"]:
            best = record
            Path(cfg["paths"]["val_best_path"]).write_text(json.dumps(best, indent=2), encoding="utf-8")
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["convergence"]["patience"]) and baseline is not None:
                break

    summary = {
        "best": best,
        "baseline": baseline,
        "n_iters": len(history),
        "wall_seconds": round(time.time() - t_start, 3),
        "human_interventions": 0,
        "history_ids": [h["id"] for h in history],
        "selection_metric": "ndcg@10",
        "val_impressions": meta.get("val_impressions"),
    }
    (results_dir / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--fresh", action="store_true", help="Wipe stale logs/val_best before this run")
    args = p.parse_args()
    print(json.dumps(run_agent(args.config, args.max_iters, fresh=args.fresh), indent=2))


if __name__ == "__main__":
    main()
