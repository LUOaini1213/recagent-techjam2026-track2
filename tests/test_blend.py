import numpy as np
import pandas as pd

from src.eval.metrics import ranking_metrics
from src.models.blend import blend_scores


def test_w1_equals_gbdt():
    g = np.array([0.9, 0.1, 0.8, 0.2])
    p = np.array([0.2, 0.9, 0.1, 0.7])
    assert np.allclose(blend_scores(g, p, 1.0), g)
    assert np.allclose(blend_scores(g, p, 0.0), p)


def test_blend_changes_order_on_short_lists():
    df = pd.DataFrame(
        {
            "eval_user_id": [1, 1, 1, 1],
            "is_click": [1, 0, 0, 0],
            "gbdt": [0.4, 0.5, 0.6, 0.7],
            "pop": [0.9, 0.2, 0.1, 0.0],
        }
    )
    df["pred"] = blend_scores(df["gbdt"].to_numpy(), df["pop"].to_numpy(), 0.0)
    pop = ranking_metrics(df)
    df["pred"] = blend_scores(df["gbdt"].to_numpy(), df["pop"].to_numpy(), 1.0)
    gbdt = ranking_metrics(df)
    assert pop["primary"] != gbdt["primary"]
    assert pop["primary"] == pop["ndcg@10"]
