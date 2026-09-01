"""Official Track 2 split, train-only vocab, CWM-style cats + item stats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.synthetic import write_synthetic_pure
from src.data.vocab import field_dims, fit_vocab, transform_col
from src.eval.metrics import impression_profile

ACTIVE = {
    "full_active": 4,
    "high_active": 3,
    "middle_active": 2,
    "2_14_day_new": 0,
    "low_active": 1,
    "single_low_active": 1,
    "30day_retention": 0,
    "day_new": 0,
    "UNKNOWN": 0,
}
FOLLOW = {"0": 0, "(0,10]": 1, "(10,50]": 2, "(50,100]": 3, "(100,150]": 4, "(150,250]": 5, "(250,500]": 6, "500+": 7}
FANS = {
    "0": 0,
    "[1,10)": 1,
    "[10,100)": 2,
    "[100,1k)": 3,
    "[1k,5k)": 4,
    "[5k,1w)": 5,
    "[1w,10w)": 6,
    "[10w,100w)": 6,
    "[100w,1000w)": 6,
}
FRIEND = {"0": 0, "[1,5)": 1, "[5,30)": 2, "[30,60)": 3, "[60,120)": 4, "[120,250)": 5, "250+": 6}
REG = {"8-14": 0, "15-30": 0, "31-60": 1, "61-90": 2, "91-180": 3, "181-365": 4, "366-730": 5, "730+": 6}
VIDEO_TYPE = {"NORMAL": 1, "AD": 0, "UNKNOWN": 0}

CAT_COLS = [
    "user_id",
    "video_id",
    "author_id",
    "music_id",
    "tag_pop",
    "upload_type",
    "tab",
    "hour",
    "duration_bucket",
    "show_bucket",
    "like_bucket",
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "video_type",
]
NUM_COLS = ["duration_ms", "show_cnt", "play_cnt", "play_progress", "like_cnt", "valid_play_cnt"]
STAT_COLS = ["show_cnt", "play_cnt", "play_progress", "like_cnt", "valid_play_cnt"]


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _map_or_zero(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).fillna(0).astype(np.int64)


def _has_real_logs(raw_dir: Path) -> bool:
    return (raw_dir / "log_standard_4_08_to_4_21_pure.csv").exists() and (
        raw_dir / "log_standard_4_22_to_5_08_pure.csv"
    ).exists()


def is_prepared(cfg: dict) -> bool:
    d = Path(cfg["paths"]["processed_dir"])
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return False
    needed = ["train.parquet", "val.parquet", "test.parquet", "vocab.json"]
    if not all((d / name).exists() for name in needed):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    has_real = _has_real_logs(Path(cfg["paths"]["raw_dir"]))
    if has_real and meta.get("used_synthetic"):
        return False
    return True


def ensure_prepared(config_path: str | Path = "configs/default.yaml", force: bool = False) -> dict:
    cfg = load_config(config_path)
    if not force and is_prepared(cfg):
        return json.loads((Path(cfg["paths"]["processed_dir"]) / "meta.json").read_text(encoding="utf-8"))
    return prepare(config_path)


def prepare(config_path: str | Path = "configs/default.yaml", allow_synthetic: bool = True) -> dict:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    used_synthetic = False
    if not _has_real_logs(raw_dir):
        if not allow_synthetic:
            raise FileNotFoundError(f"KuaiRand-Pure logs not found in {raw_dir}")
        write_synthetic_pure(raw_dir)
        used_synthetic = True

    early = pd.read_csv(raw_dir / "log_standard_4_08_to_4_21_pure.csv")
    late = pd.read_csv(raw_dir / "log_standard_4_22_to_5_08_pure.csv")
    late = late.sort_values("time_ms", kind="mergesort").reset_index(drop=True)
    cut = int(len(late) * cfg["split"]["val_fraction"])
    val_raw, test_raw, train_raw = late.iloc[:cut].copy(), late.iloc[cut:].copy(), early.copy()

    user_fe = _read_optional(raw_dir / "user_features_pure.csv")
    video_fe = _read_optional(raw_dir / "video_features_basic_pure.csv")
    video_stat = _read_optional(raw_dir / "video_features_statistic_pure.csv")

    frames = {
        name: _featurize(df, user_fe, video_fe, video_stat)
        for name, df in [("train", train_raw), ("val", val_raw), ("test", test_raw)]
    }

    cat_cols = [c for c in CAT_COLS if c in frames["train"].columns]
    vocabs = fit_vocab(frames["train"], cat_cols)
    dims = field_dims(vocabs)
    for name, df in frames.items():
        encoded = df.copy()
        encoded["eval_user_id"] = df["user_id"].to_numpy()
        for col in cat_cols:
            encoded[col] = transform_col(df[col], vocabs[col])
        for col in NUM_COLS:
            if col in encoded.columns:
                encoded[col] = pd.to_numeric(encoded[col], errors="coerce").fillna(0.0)
            else:
                encoded[col] = 0.0
        encoded.to_parquet(out_dir / f"{name}.parquet", index=False)
        frames[name] = encoded

    val_profile = impression_profile(frames["val"])
    meta = {
        "field_dims": dims,
        "cat_cols": cat_cols,
        "num_cols": [c for c in NUM_COLS if c in frames["train"].columns],
        "used_synthetic": used_synthetic,
        "n_train": int(len(frames["train"])),
        "n_val": int(len(frames["val"])),
        "n_test": int(len(frames["test"])),
        "click_rate_train": float(frames["train"]["is_click"].mean()),
        "val_impressions": val_profile,
        "note": "Categorical vocab is fit on train only. Index 0 is OOV. Test labels must not be used until scripts/final_eval_test.py.",
    }
    (out_dir / "vocab.json").write_text(json.dumps(vocabs), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _featurize(
    df: pd.DataFrame,
    user_fe: pd.DataFrame | None,
    video_fe: pd.DataFrame | None,
    video_stat: pd.DataFrame | None,
) -> pd.DataFrame:
    out = df.copy()
    if "hourmin" in out.columns:
        out["hour"] = (out["hourmin"].fillna(0).astype(np.int64) // 100) % 24
    else:
        out["hour"] = 0
    duration = pd.to_numeric(out.get("duration_ms", 0), errors="coerce").fillna(0.0)
    out["duration_ms"] = duration
    out["duration_bucket"] = np.clip((duration / 1000.0 / 10.0).astype(int), 0, 40)

    if user_fe is not None:
        uf = user_fe.copy()
        if "user_active_degree" in uf.columns:
            uf["user_active_degree"] = _map_or_zero(uf["user_active_degree"], ACTIVE)
        for col, mapping in [
            ("follow_user_num_range", FOLLOW),
            ("fans_user_num_range", FANS),
            ("friend_user_num_range", FRIEND),
            ("register_days_range", REG),
        ]:
            if col in uf.columns:
                uf[col] = _map_or_zero(uf[col], mapping)
        keep = [c for c in ["user_id", "user_active_degree", "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range"] if c in uf.columns]
        out = out.merge(uf[keep], on="user_id", how="left")

    if video_fe is not None:
        vf = video_fe.copy()
        if "video_type" in vf.columns:
            vf["video_type"] = _map_or_zero(vf["video_type"], VIDEO_TYPE)
        if "upload_type" in vf.columns:
            vf["upload_type"] = vf["upload_type"].fillna("UNKNOWN").astype(str)
        if "tag" in vf.columns:
            vf["tag_ls"] = vf["tag"].astype(str).str.split(",")
            counts: dict[str, int] = {}
            for ls in vf["tag_ls"]:
                for t in ls:
                    counts[t] = counts.get(t, 0) + 1
            vf["tag_pop"] = vf["tag_ls"].apply(lambda ls: max(ls, key=lambda t: counts.get(t, 0)))
        keep_v = [c for c in ["video_id", "author_id", "music_id", "video_type", "upload_type", "tag_pop"] if c in vf.columns]
        out = out.merge(vf[keep_v], on="video_id", how="left", suffixes=("", "_vf"))
        if "author_id_vf" in out.columns:
            out["author_id"] = out["author_id"].fillna(out["author_id_vf"]) if "author_id" in out.columns else out["author_id_vf"]

    if video_stat is not None:
        keep_s = [c for c in ["video_id", *STAT_COLS] if c in video_stat.columns]
        stat = video_stat[keep_s].groupby("video_id", as_index=False).mean(numeric_only=True)
        out = out.merge(stat, on="video_id", how="left")

    for col in STAT_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["show_bucket"] = np.clip(np.log1p(out["show_cnt"]).astype(int), 0, 16)
    out["like_bucket"] = np.clip(np.log1p(out["like_cnt"]).astype(int), 0, 12)

    for col in ["author_id", "music_id", "tag_pop", "upload_type", "tab", "user_active_degree", "video_type"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0)

    for col in ["long_view", "is_like"]:
        if col not in out.columns:
            out[col] = 0
    out["is_click"] = out["is_click"].fillna(0).astype(np.int64)
    return out


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, default=str))
