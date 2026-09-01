"""Zip the repo for another Grok bot. Excludes caches and the duplicate tar.gz."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "recagent-track2-for-grok-bot.zip"

SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", ".idea", ".vscode"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_FILES = {
    "KuaiRand-Pure.tar.gz",
    "recagent-track2-for-grok-bot.zip",
    "log_random_4_22_to_5_08_pure.csv",  # unused by current train/eval
}


def keep(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_DIR_NAMES for part in rel.parts):
        return False
    if path.name in SKIP_FILES:
        return False
    if path.suffix in SKIP_SUFFIXES:
        return False
    return True


def main() -> None:
    files = [p for p in ROOT.rglob("*") if p.is_file() and keep(p)]
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            zf.write(p, p.relative_to(ROOT).as_posix())
    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({mb:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
