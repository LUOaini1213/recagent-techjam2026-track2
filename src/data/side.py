"""Train-safe KuaiRand side tables. No val/test click labels, no current-row engagement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

USER_CAT = [
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
] + [f"onehot_feat{i}" for i in range(18)]
USER_NUM = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
VIDEO_NUM = [
    "complete_play_cnt",
    "long_time_play_cnt",
    "short_time_play_cnt",
    "share_cnt",
    "collect_cnt",
    "follow_cnt",
    "download_cnt",
    "comment_cnt",
    "play_user_num",
    "like_user_num",
    "share_user_num",
    "collect_user_num",
]
VIDEO_CAT = ["music_type"]
LEAK_COLS = {
    "play_time_ms",
    "is_like",
    "long_view",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
}


def side_feature_names() -> list[str]:
    return USER_CAT + USER_NUM + VIDEO_NUM + VIDEO_CAT


def add_side_features(
    df: pd.DataFrame,
    raw_dir: Path,
    vocab_path: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join static user/video columns. Click labels on `df` are ignored."""
    out = df.copy()
    user_key = "eval_user_id" if "eval_user_id" in out.columns else "user_id"
    users = pd.read_csv(raw_dir / "user_features_pure.csv")
    keep_u = ["user_id"] + [c for c in USER_CAT + USER_NUM if c in users.columns]
    users = users[keep_u].drop_duplicates("user_id")
    users = users.rename(columns={"user_id": user_key})
    out = out.merge(users, on=user_key, how="left")

    vocabs = json.loads(vocab_path.read_text(encoding="utf-8"))
    inv_video = {int(v): k for k, v in vocabs["video_id"].items()}
    out["_orig_video"] = out["video_id"].map(inv_video)

    stats = pd.read_csv(raw_dir / "video_features_statistic_pure.csv")
    keep_s = ["video_id"] + [c for c in VIDEO_NUM if c in stats.columns]
    stats = stats[keep_s].groupby("video_id", as_index=False).mean(numeric_only=True)
    stats = stats.rename(columns={"video_id": "_orig_video"})
    stats["_orig_video"] = stats["_orig_video"].astype(str)
    out = out.merge(stats, on="_orig_video", how="left")

    basic = pd.read_csv(raw_dir / "video_features_basic_pure.csv")
    if "music_type" in basic.columns:
        b = basic[["video_id", "music_type"]].drop_duplicates("video_id")
        b = b.rename(columns={"video_id": "_orig_video"})
        b["_orig_video"] = b["_orig_video"].astype(str)
        out = out.merge(b, on="_orig_video", how="left")

    for c in USER_CAT + VIDEO_CAT:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(lower=0).astype(np.int64)
    for c in USER_NUM + VIDEO_NUM:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = np.log1p(pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0))
    drop = [c for c in out.columns if c.startswith("_orig") or c.endswith("_side") or c.endswith("_mt") or c == "video_id_y"]
    out = out.drop(columns=[c for c in drop if c in out.columns], errors="ignore")
    if "video_id_x" in out.columns:
        out = out.rename(columns={"video_id_x": "video_id"})
    cats = [c for c in USER_CAT + VIDEO_CAT if c in out.columns]
    nums = [c for c in USER_NUM + VIDEO_NUM if c in out.columns]
    assert not (LEAK_COLS & set(cats + nums))
    return out, cats, nums
