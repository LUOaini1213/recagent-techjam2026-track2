import pandas as pd

from src.eval.metrics import impression_profile, ndcg_at_k, ranking_metrics, recall_at_k


def test_ndcg_perfect():
    assert abs(ndcg_at_k([1, 1, 0, 0], 10) - 1.0) < 1e-6


def test_recall():
    assert abs(recall_at_k([1, 0, 1, 0], 2) - 0.5) < 1e-6


def test_macro_skips_users_without_click():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "is_click": [1, 0, 0, 0, 0, 1],
            "pred": [0.9, 0.2, 0.1, 0.8, 0.1, 0.7],
        }
    )
    m = ranking_metrics(df)
    assert m["n_users"] == 3
    assert m["n_scored_users"] == 2
    assert m["ndcg@10"] > 0.9
    assert m["primary"] == m["ndcg@10"]


def test_recall50_saturates_on_short_lists():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "is_click": [0, 1, 0, 1],
            "pred": [0.1, 0.2, 0.3, 0.4],
        }
    )
    m = ranking_metrics(df)
    assert m["recall@50"] == 1.0
    assert m["impressions"]["frac_list_le_50"] == 1.0
    assert m["primary"] < 1.0


def test_impression_profile():
    df = pd.DataFrame({"user_id": [1, 1, 2, 2, 2]})
    p = impression_profile(df)
    assert p["n_users"] == 2
    assert p["list_len_p50"] == 2.5


def test_eval_user_id_not_collapsed_by_oov_code():
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0],
            "eval_user_id": [10, 10, 11, 11],
            "is_click": [1, 0, 1, 0],
            "pred": [0.9, 0.1, 0.8, 0.2],
        }
    )
    m = ranking_metrics(df)
    assert m["n_users"] == 2
    assert m["n_scored_users"] == 2
    assert m["primary"] == m["ndcg@10"]
    assert m["recall@50"] == 1.0
