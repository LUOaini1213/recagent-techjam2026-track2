"""In-list-varying video/session columns. No click labels, no current-row engagement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.side import LEAK_COLS, add_side_features

INLIST_CAT = ["tag_a", "tag_b"]
INLIST_NUM = ["n_tags", "server_width", "server_height", "aspect", "time_rank", "time_z"]


def add_inlist_features(
    df: pd.DataFrame,
    raw_dir: Path,
    vocab_path: Path,
    mode: str = "all",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out, side_cat, side_num = add_side_features(df, raw_dir, vocab_path)
    vocabs = json.loads(vocab_path.read_text(encoding="utf-8"))
    inv_video = {int(v): k for k, v in vocabs["video_id"].items()}
    orig = out["video_id"].map(inv_video).astype(str)

    basic = pd.read_csv(raw_dir / "video_features_basic_pure.csv")
    keep = [c for c in ("video_id", "tag", "server_width", "server_height") if c in basic.columns]
    basic = basic[keep].drop_duplicates("video_id")
    basic["video_id"] = basic["video_id"].astype(str)
    tmp = pd.DataFrame({"_orig_video": orig.to_numpy()}, index=out.index)
    tmp = tmp.merge(basic.rename(columns={"video_id": "_orig_video"}), on="_orig_video", how="left")
    tmp.index = out.index

    tags = tmp["tag"].fillna("").astype(str).str.split(",")
    out["n_tags"] = tags.map(lambda xs: float(len([t for t in xs if t and t != "nan"])))
    out["tag_a"] = pd.to_numeric(tags.map(lambda xs: xs[0] if xs and xs[0] not in ("", "nan") else 0), errors="coerce").fillna(0).astype(np.int64)
    out["tag_b"] = pd.to_numeric(tags.map(lambda xs: xs[1] if len(xs) > 1 and xs[1] not in ("", "nan") else 0), errors="coerce").fillna(0).astype(np.int64)
    out["server_width"] = pd.to_numeric(tmp.get("server_width"), errors="coerce").fillna(0.0)
    out["server_height"] = pd.to_numeric(tmp.get("server_height"), errors="coerce").fillna(0.0)
    out["aspect"] = np.where(out["server_height"] > 0, out["server_width"] / out["server_height"], 0.0)

    uc = "eval_user_id" if "eval_user_id" in out.columns else "user_id"
    if "time_ms" in out.columns:
        out["time_rank"] = out.groupby(uc, sort=False)["time_ms"].rank(method="first")
        mu = out.groupby(uc, sort=False)["time_ms"].transform("mean")
        sd = out.groupby(uc, sort=False)["time_ms"].transform("std")
        sd = sd.replace(0, 1.0).fillna(1.0)
        out["time_z"] = (out["time_ms"] - mu) / sd
    else:
        out["time_rank"] = 0.0
        out["time_z"] = 0.0

    extra_cat = list(side_cat)
    extra_num = list(side_num)
    if mode in ("all", "tags"):
        extra_cat += [c for c in ("tag_a", "tag_b") if c in out.columns]
        extra_num += [c for c in ("n_tags",) if c in out.columns]
    if mode in ("all", "res"):
        extra_num += [c for c in ("server_width", "server_height", "aspect") if c in out.columns]
    if mode in ("all", "time"):
        extra_num += [c for c in ("time_rank", "time_z") if c in out.columns]
    assert not (LEAK_COLS & set(extra_cat + extra_num))
    return out, extra_cat, extra_num
