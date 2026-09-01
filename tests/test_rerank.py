import pandas as pd

from src.eval.metrics import ranking_metrics
from src.models.rerank import search_rerank, user_z


def test_user_z_is_zero_mean_per_user():
    df = pd.DataFrame({"eval_user_id": [1, 1, 1, 2, 2], "s": [1.0, 2.0, 3.0, 10.0, 12.0]})
    z = user_z(df, "s", "eval_user_id")
    df["z"] = z
    means = df.groupby("eval_user_id")["z"].mean()
    assert abs(float(means.loc[1])) < 1e-9
    assert abs(float(means.loc[2])) < 1e-9


def test_search_rerank_uses_primary_ndcg_and_eval_user_id():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0],
            "eval_user_id": [10, 10, 11, 11],
            "is_click": [1, 0, 1, 0],
            "gbdt": [0.6, 0.4, 0.55, 0.45],
            "pop": [0.1, 0.9, 0.2, 0.8],
            "time_ms": [1, 2, 1, 2],
        }
    )
    best = search_rerank(df)
    assert best["n_users"] == 2
    assert best["primary"] == best["ndcg@10"]
    assert best["primary"] >= ranking_metrics(df.assign(pred=df["gbdt"]))["primary"] - 1e-12
