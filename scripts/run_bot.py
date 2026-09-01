"""One entry for a Grok bot: real data, fresh logs, full agent catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import os

    os.chdir(ROOT)
    os.environ.setdefault("PYTHONPATH", str(ROOT))

    from src.data.prepare import ensure_prepared, load_config
    from agent.loop import run_agent

    cfg = load_config(ROOT / "configs" / "default.yaml")
    meta = ensure_prepared(str(ROOT / "configs" / "default.yaml"))
    if meta.get("used_synthetic"):
        raise SystemExit(
            "Processed data is synthetic. Run: python scripts/download_kuairand.py && python -m src.data.prepare"
        )
    n_val = int(meta.get("n_val") or 0)
    if n_val < 10_000:
        raise SystemExit(f"val too small ({n_val}); refusing to start. Re-run prepare on KuaiRand-Pure.")

    print(json.dumps({"prepared": True, "meta_summary": {
        "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"),
        "used_synthetic": meta.get("used_synthetic"),
        "val_impressions": meta.get("val_impressions"),
    }}, indent=2))
    summary = run_agent(str(ROOT / "configs" / "default.yaml"), fresh=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
