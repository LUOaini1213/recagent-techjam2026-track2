"""Download KuaiRand-Pure from Zenodo (md5 0820331067a3784d9691136f772b35a7)."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
MD5 = "0820331067a3784d9691136f772b35a7"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest_root: Path = Path("data/raw")) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    archive = dest_root / "KuaiRand-Pure.tar.gz"
    if not archive.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    digest = md5sum(archive)
    if digest != MD5:
        raise RuntimeError(f"md5 mismatch: {digest} != {MD5}")
    print("Extracting...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_root)
    return dest_root / "KuaiRand-Pure"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dest", default="data/raw")
    args = p.parse_args()
    print(download(Path(args.dest)))
