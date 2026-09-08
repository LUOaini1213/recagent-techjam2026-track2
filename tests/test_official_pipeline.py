"""Tests for the scored pipeline in scripts/official_*.py.

They run on synthetic frames — no dataset, no trained models — and pin down
the three claims the README makes about the pipeline: the vectorized
early-stopping metric equals the official evaluator, train-row statistics in
past mode use strictly earlier days only, and the final blend is the recipe
final.json says it is.
"""
import os

import numpy as np
import pandas as pd
import pytest

from evaluate import evaluate  # kuairand-starter-kit, via pytest.ini pythonpath
import official_final
from official_blend import rank01, rank_within
from official_lgbm import OfficialFeval, _train_sc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _random_users(seed=0, n_users=300, positive_rate=0.35):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 12, size=n_users)
    users = np.repeat(np.arange(n_users), sizes)  # user-contiguous, as the cache is
    y = (rng.random(len(users)) < positive_rate).astype(np.float64)
    s = rng.random(len(users))  # continuous scores: no ties
    return users, y, s


def test_vectorized_feval_equals_the_official_evaluator():
    users, y, s = _random_users()
    fe = OfficialFeval(users)
    official = evaluate(users.tolist(), y.tolist(), s.tolist())
    assert abs(fe.primary(y, s) - official['primary']) < 1e-9


def test_feval_handles_users_without_positives_like_the_protocol():
    # A user with no positive label contributes nDCG = 0 and is skipped by GAUC.
    users = np.array([0, 0, 0, 1, 1, 2, 2])
    y = np.array([0, 0, 0, 1, 0, 1, 1], dtype=np.float64)
    s = np.array([0.9, 0.5, 0.1, 0.8, 0.2, 0.7, 0.3])
    fe = OfficialFeval(users)
    official = evaluate(users.tolist(), y.tolist(), s.tolist())
    assert abs(fe.primary(y, s) - official['primary']) < 1e-12
    # user 1 is ranked perfectly, user 0 has no positive (nDCG 0), user 2 is all-positive (nDCG 1, no GAUC)
    assert official['GAUC'] == 1.0
    assert abs(official['nDCG@5'] - (0.0 + 1.0 + 1.0) / 3) < 1e-12


def test_past_mode_train_statistics_use_strictly_earlier_days_only():
    tr = pd.DataFrame({
        'item': ['a', 'a', 'a', 'a', 'b', 'b'],
        'date': [1, 1, 2, 3, 1, 2],
        'y':    [1, 0, 1, 0, 1, 1],
    })
    s, c = _train_sc(tr, ['item'], 'y', past=True)
    # item a: day 1 rows see nothing; day 2 sees day 1 (sum 1, count 2); day 3 sees days 1-2 (sum 2, count 3)
    assert s.tolist() == [0, 0, 1, 2, 0, 1]
    assert c.tolist() == [0, 0, 2, 3, 0, 1]


def test_leave_one_out_mode_removes_only_the_row_itself():
    tr = pd.DataFrame({'item': ['a', 'a', 'a', 'b'], 'date': [1, 2, 3, 1], 'y': [1, 0, 1, 1]})
    s, c = _train_sc(tr, ['item'], 'y', past=False)
    assert s.tolist() == [1, 2, 1, 0]
    assert c.tolist() == [2, 2, 2, 0]


def test_rank01_is_a_monotone_map_onto_the_unit_interval():
    x = np.array([3.0, -1.0, 7.5, 0.0, 2.0])
    r = rank01(x)
    assert r.min() == 0.0 and r.max() == 1.0
    order = np.argsort(x)
    assert np.all(np.diff(r[order]) > 0)
    assert np.array_equal(rank01(x), official_final.rank01(x))


def test_rank_within_user_ignores_other_users_scores():
    uid = np.array([0, 0, 0, 1, 1, 2])
    x = np.array([10.0, 30.0, 20.0, 0.1, 0.2, 5.0])
    r = rank_within(x, uid)
    assert r[:3].tolist() == [0.0, 1.0, 0.5]
    assert r[3:5].tolist() == [0.0, 1.0]
    assert r[5] == 0.0  # a single-impression user has no within-user order


def test_final_blend_is_the_documented_weighted_rank_average():
    rng = np.random.default_rng(1)
    a, b = rng.random(50), rng.random(50)
    out = official_final.blend({'gbdt5v2': a, 'rank3': b})
    expected = 0.8 * rank01(a) + 0.2 * rank01(b)
    assert np.allclose(out, expected)
    same = official_final.blend({'gbdt5v2': a, 'rank3': a})
    assert np.allclose(same, rank01(a))


def test_committed_final_artifacts_reproduce_from_the_group_predictions():
    preds = os.path.join(ROOT, 'results', 'official', 'preds')
    needed = [f'{g}_{s}.npy' for g in ('gbdt5v2', 'rank3', 'FINAL') for s in ('valid', 'test')]
    if not all(os.path.exists(os.path.join(preds, n)) for n in needed):
        pytest.skip('committed prediction artifacts not present')
    assert official_final.check() == 0


def test_bootstrap_per_user_pieces_reproduce_the_official_evaluator():
    # scripts/official_bootstrap.py resamples per-user pieces; the aggregate of
    # those pieces on the full sample must equal evaluate.py exactly, including
    # users with no positives (nDCG 0, skipped by GAUC) and all-positive users.
    import official_bootstrap as ob
    users, y, s = _random_users(seed=3, n_users=400, positive_rate=0.33)
    npos, auc, ndcg = ob.per_user(users, y, s)
    gauc, nd, primary = ob.aggregate(npos, auc, ndcg, np.arange(len(npos)))
    official = evaluate(users.tolist(), y.tolist(), s.tolist())
    assert abs(gauc - official['GAUC']) < 1e-9
    assert abs(nd - official['nDCG@5']) < 1e-9
    assert abs(primary - official['primary']) < 1e-9


def test_bootstrap_resample_is_paired_and_order_free():
    import official_bootstrap as ob
    users, y, s = _random_users(seed=4, n_users=200)
    pieces = ob.per_user(users, y, s)
    full = ob.aggregate(*pieces, np.arange(200))
    perm = np.random.default_rng(0).permutation(200)
    assert abs(ob.aggregate(*pieces, perm)[2] - full[2]) < 1e-12  # same users, any order
    dup = np.repeat(np.arange(200), 2)
    assert abs(ob.aggregate(*pieces, dup)[2] - full[2]) < 1e-12   # duplicating every user is a no-op
