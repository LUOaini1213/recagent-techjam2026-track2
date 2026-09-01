"""Tiny KuaiRand-shaped tables so the pipeline can run without Zenodo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_synthetic_pure(raw_dir: Path, n_users: int = 40, n_items: int = 80, seed: int = 61) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    users = np.arange(n_users)
    items = np.arange(n_items)
    item_pop = rng.random(n_items)
    item_pop = item_pop / item_pop.sum()

    rows = []
    t0 = 1_650_000_000_000
    for split, date0, n_rows in [
        ("early", 20220415, 4000),
        ("late", 20220428, 4000),
    ]:
        for i in range(n_rows):
            u = int(rng.integers(0, n_users))
            v = int(rng.choice(items, p=item_pop))
            hour = int(rng.integers(0, 24))
            duration = int(rng.integers(6_000, 60_000))
            click = int(rng.random() < (0.08 + 0.25 * item_pop[v] + 0.05 * ((u + v) % 5 == 0)))
            play = int(duration * (0.2 + 0.8 * click) * rng.random())
            rows.append(
                {
                    "user_id": u,
                    "video_id": v,
                    "date": date0,
                    "hourmin": hour * 100,
                    "time_ms": t0 + i * 1000 + (0 if split == "early" else 10_000_000),
                    "is_click": click,
                    "is_like": int(click and rng.random() < 0.2),
                    "is_follow": 0,
                    "is_comment": 0,
                    "is_forward": 0,
                    "is_hate": 0,
                    "long_view": int(play > 18_000),
                    "play_time_ms": play,
                    "duration_ms": duration,
                    "tab": int(rng.integers(0, 3)),
                    "author_id": v % 20,
                    "music_id": v % 15,
                    "video_type": "NORMAL",
                    "upload_type": "ShortImport",
                    "tag": str(v % 8),
                }
            )
    df = pd.DataFrame(rows)
    early = df[df["date"] == 20220415]
    late = df[df["date"] == 20220428]
    early.to_csv(raw_dir / "log_standard_4_08_to_4_21_pure.csv", index=False)
    late.to_csv(raw_dir / "log_standard_4_22_to_5_08_pure.csv", index=False)

    user_fe = pd.DataFrame(
        {
            "user_id": users,
            "user_active_degree": rng.choice(
                ["full_active", "high_active", "middle_active", "low_active"], n_users
            ),
            "is_lowactive_period": 0,
            "follow_user_num_range": rng.choice(["0", "(0,10]", "(10,50]"], n_users),
            "fans_user_num_range": rng.choice(["0", "[1,10)", "[10,100)"], n_users),
            "friend_user_num_range": rng.choice(["0", "[1,5)"], n_users),
            "register_days_range": rng.choice(["91-180", "181-365", "730+"], n_users),
        }
    )
    user_fe.to_csv(raw_dir / "user_features_pure.csv", index=False)

    video_fe = pd.DataFrame(
        {
            "video_id": items,
            "author_id": items % 20,
            "video_type": "NORMAL",
            "upload_type": "ShortImport",
            "tag": [str(i % 8) for i in items],
            "music_id": items % 15,
        }
    )
    video_fe.to_csv(raw_dir / "video_features_basic_pure.csv", index=False)
    return raw_dir
