"""Paired user-level bootstrap of (final - FM) on the hidden test set.

    python scripts/official_bootstrap.py     # writes results/official/bootstrap.json

Is +0.0069 more than noise? The official metric is a per-user aggregate, so
the right resampling unit is the user: draw users with replacement, recompute
primary for BOTH systems on the same draw, and look at the distribution of the
difference. Per-user contributions are computed once and checked against the
official evaluator on the full sample before any resampling.

Inputs (all produced by the pipeline; nothing here touches test labels for
any decision — the submission was frozen on 2026-09-01):
  labels  data/interim/official_feats_past/test.parquet  (user-contiguous), or
          the committed results/official/labels_valid_test.npz, same row order
  ours    results/official/preds/FINAL_test.npy
  FM      results/official/fm_feats/test.npy   (our torch-FM reproduction,
          test primary 0.5943 vs the published 0.5946). Not committed, so
          rerunning this script needs the dataset and scripts/official_fm.py;
          tests/test_readme_numbers.py only ties bootstrap.json to final.json
          and the README.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate  # noqa: E402  official scorer, unmodified
import official_labels  # noqa: E402  (committed labels + their digest guard)

OUT = os.path.join(ROOT, 'results', 'official', 'bootstrap.json')
REPS, SEED = 2000, 0


def per_user(uid, y, s):
    """Per-user pieces the official metric aggregates from.

    GAUC  = sum_u npos_u * auc_u / sum_u npos_u   over users with 0 < npos < n
    nDCG5 = mean_u ndcg_u                          over ALL users (0 if npos == 0)
    """
    codes, counts = np.unique(uid, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    npos = np.zeros(len(codes))
    auc = np.full(len(codes), np.nan)
    ndcg = np.zeros(len(codes))
    disc = 1.0 / np.log2(np.arange(2, 7))
    for i, (st, n) in enumerate(zip(starts, counts)):
        yy, ss = y[st:st + n], s[st:st + n]
        p = yy.sum()
        npos[i] = p
        if 0 < p < n:
            r = rankdata(ss)                       # average ranks on ties
            auc[i] = (r[yy == 1].sum() - p * (p + 1) / 2) / (p * (n - p))
        if p > 0:
            order = np.argsort(-ss, kind='stable')
            k = min(5, n)
            dcg = (yy[order][:k] * disc[:k]).sum()
            idcg = disc[:int(min(p, 5))].sum()
            ndcg[i] = dcg / idcg
    return npos, auc, ndcg


def aggregate(npos, auc, ndcg, idx):
    n_, a_, d_ = npos[idx], auc[idx], ndcg[idx]
    ok = ~np.isnan(a_)
    gauc = (n_[ok] * a_[ok]).sum() / n_[ok].sum()
    return gauc, d_.mean(), 0.5 * (gauc + d_.mean())


def main():
    cache = os.path.join(ROOT, 'data', 'interim', 'official_feats_past', 'test.parquet')
    if os.path.exists(cache):
        te = pd.read_parquet(cache, columns=['user_id', 'long_view'])
        uid = te['user_id'].to_numpy()
        y = te['long_view'].to_numpy().astype(np.float64)
    else:
        uid, y = official_labels.split_labels('test')
        if uid is None:
            raise SystemExit('no test labels: build the feature cache or restore '
                             'results/official/labels_valid_test.npz')
        y = y.astype(np.float64)
    ours = np.load(os.path.join(ROOT, 'results', 'official', 'preds', 'FINAL_test.npy'))
    fm_path = os.path.join(ROOT, 'results', 'official', 'fm_feats', 'test.npy')
    if not os.path.exists(fm_path):
        raise SystemExit(f'{os.path.relpath(fm_path, ROOT)} is not committed: run '
                         'scripts/download_kuairand.py, then scripts/official_fm.py')
    fm = np.load(fm_path)

    pieces = {}
    for name, s in (('final', ours), ('fm', fm)):
        npos, auc, ndcg = per_user(uid, y, s)
        full = aggregate(npos, auc, ndcg, np.arange(len(npos)))
        off = evaluate(uid.tolist(), y.tolist(), s.tolist())
        assert abs(full[2] - off['primary']) < 1e-9, (name, full[2], off['primary'])
        pieces[name] = (npos, auc, ndcg)
        print(f"{name:5s} full-sample primary {full[2]:.6f}  "
              f"(official evaluate.py: {off['primary']:.6f})  users {len(npos)}")

    n_users = len(pieces['final'][0])
    rng = np.random.default_rng(SEED)
    d_primary, d_gauc, d_ndcg = [], [], []
    for _ in range(REPS):
        idx = rng.integers(0, n_users, n_users)
        g1, n1, p1 = aggregate(*pieces['final'], idx)
        g0, n0, p0 = aggregate(*pieces['fm'], idx)
        d_primary.append(p1 - p0)
        d_gauc.append(g1 - g0)
        d_ndcg.append(n1 - n0)
    d_primary, d_gauc, d_ndcg = map(np.asarray, (d_primary, d_gauc, d_ndcg))

    def summ(d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {'mean': round(float(d.mean()), 5),
                'ci95': [round(float(lo), 5), round(float(hi), 5)],
                'p_delta_le_0': round(float((d <= 0).mean()), 4)}

    res = {
        'method': f'paired user-level bootstrap, {REPS} resamples of {n_users} '
                  'test users with replacement, seed 0; per-user pieces checked '
                  'against the official evaluate.py on the full sample',
        'point_estimate': {
            'final_primary': round(float(aggregate(*pieces['final'], np.arange(n_users))[2]), 6),
            'fm_repro_primary': round(float(aggregate(*pieces['fm'], np.arange(n_users))[2]), 6),
            'fm_published_primary': 0.5946},
        'delta_final_minus_fm_repro': {
            'primary': summ(d_primary), 'GAUC': summ(d_gauc), 'nDCG@5': summ(d_ndcg)},
    }
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    dp = res['delta_final_minus_fm_repro']['primary']
    print(f"\ndelta primary (final - FM): mean {dp['mean']:+.4f}  "
          f"95% CI [{dp['ci95'][0]:+.4f}, {dp['ci95'][1]:+.4f}]  "
          f"P(delta <= 0) = {dp['p_delta_le_0']}")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == '__main__':
    main()
