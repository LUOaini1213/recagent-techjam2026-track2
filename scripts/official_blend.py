"""Greedy rank-average blending of saved model predictions (official metric).

python scripts/official_blend.py --models past_reg_mid,p_lr01,fm --out blend1
  fm = the raw torch-FM valid/test scores in results/official/fm_feats/

Greedy forward selection: start from the best single model, then repeatedly
try adding any model at weights 0.05..0.95 and keep the best improvement on
valid primary. Recipe saved to results/official/<out>_blend.json.
--eval_test applies the frozen recipe to test predictions (final step only).
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate
from official_lgbm import CACHE, OUT, OfficialFeval


def load_preds(model, split):
    if model == 'fm' or model.startswith('fm_'):
        return np.load(os.path.join(OUT, 'fm_feats' + model[2:], f'{split}.npy'))
    return np.load(os.path.join(OUT, 'preds', f'{model}_{split}.npy'))


def rank01(x):
    r = np.argsort(np.argsort(x, kind='stable'), kind='stable')
    return (r / (len(r) - 1)).astype(np.float64)


def rank_within(x, uid):
    """Rank inside each user's list, scaled to [0,1]. The metric only sees
    within-user order, so blending in this space stops one model's global
    score distribution from distorting another's per-user ordering.
    Rows are user-contiguous in the feature cache."""
    codes = pd.factorize(uid)[0]
    n = len(codes)
    counts = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    order = np.lexsort((x, codes))
    pos = np.arange(n) - starts[codes[order]]
    out = np.empty(n, dtype=np.float64)
    denom = np.maximum(counts[codes[order]] - 1, 1)
    out[order] = pos / denom
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', default='blend')
    ap.add_argument('--eval_test', action='store_true')
    ap.add_argument('--avg', action='store_true',
                    help='equal-weight rank average instead of greedy')
    ap.add_argument('--peruser', action='store_true',
                    help='rank within each user list instead of globally')
    a = ap.parse_args()
    models = a.models.split(',')

    va = pd.read_parquet(os.path.join(CACHE, 'valid.parquet'),
                         columns=['user_id', 'long_view'])
    fe = OfficialFeval(va['user_id'].to_numpy())
    y = va['long_view'].to_numpy().astype(np.float64)
    tf = ((lambda p: rank_within(p, va['user_id'].to_numpy())) if a.peruser
          else rank01)
    ranks = {m: tf(load_preds(m, 'valid')) for m in models}
    singles = {m: fe.primary(y, ranks[m]) for m in models}
    for m, v in sorted(singles.items(), key=lambda x: -x[1]):
        print(f"  single {m:16s} {v:.4f}")

    if a.avg:
        weights = {m: 1.0 / len(models) for m in models}
        cur = sum(ranks[m] for m in models) / len(models)
        cur_score = fe.primary(y, cur)
        print(f"  equal-weight avg -> {cur_score:.4f}")
        improved = False
    else:
        best_m = max(singles, key=singles.get)
        weights = {best_m: 1.0}
        cur = ranks[best_m].copy()
        cur_score = singles[best_m]
        improved = True
    while improved:
        improved = False
        for m in models:
            for w in np.arange(0.05, 1.0, 0.05):
                cand = (1 - w) * cur + w * ranks[m]
                sc = fe.primary(y, cand)
                if sc > cur_score + 1e-5:
                    cur_score, best_add, best_w = sc, m, w
                    improved = True
        if improved:
            cur = (1 - best_w) * cur + best_w * ranks[best_add]
            weights = {k: v * (1 - best_w) for k, v in weights.items()}
            weights[best_add] = weights.get(best_add, 0.0) + best_w
            print(f"  + {best_add} w={best_w:.2f} -> {cur_score:.4f}")

    r = evaluate(va['user_id'].tolist(), va['long_view'].tolist(), cur.tolist())
    print(f"blend valid: GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
          f"primary {r['primary']:.4f}")
    res = {'models': models, 'weights': weights,
           'valid': {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in r.items()}}
    if a.eval_test:
        te = pd.read_parquet(os.path.join(CACHE, 'test.parquet'),
                             columns=['user_id', 'long_view'])
        blend_t = np.zeros(len(te))
        for m, w in weights.items():
            blend_t += w * rank01(load_preds(m, 'test'))
        rt = evaluate(te['user_id'].tolist(), te['long_view'].tolist(),
                      blend_t.tolist())
        print(f"blend test:  GAUC {rt['GAUC']:.4f} nDCG@5 {rt['nDCG@5']:.4f} "
              f"primary {rt['primary']:.4f}")
        res['test'] = {k: round(v, 6) if isinstance(v, float) else v
                       for k, v in rt.items()}
    with open(os.path.join(OUT, f'{a.out}_blend.json'), 'w') as fh:
        json.dump(res, fh, indent=2)


if __name__ == '__main__':
    main()
