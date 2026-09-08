"""Compute the attainable range of the official metric on this benchmark.

    python scripts/official_ceiling.py      # writes results/official/ceiling.json

"Label-perfect" scores every impression by its own label — the best any
ranker can do — and "random" scores with uniform noise (mean of 5 seeds). The
README quotes these to put +0.0069 on a scale: the ceiling is well below 1.0
because users with no positive label contribute nDCG = 0 by protocol and users
with all-positive labels are excluded from GAUC.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
from evaluate import evaluate  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'official', 'ceiling.json')
CACHES = [os.path.join(ROOT, 'data', 'interim', 'official_feats_past'),
          os.path.join(ROOT, 'data', 'interim', 'official_feats')]


def labels(split):
    for cache in CACHES:
        p = os.path.join(cache, f'{split}.parquet')
        if os.path.exists(p):
            df = pd.read_parquet(p, columns=['user_id', 'long_view'])
            return df['user_id'].to_numpy(), df['long_view'].to_numpy().astype(np.float64)
    raise SystemExit('feature cache not found: run scripts/download_kuairand.py, '
                     'then scripts/official_lgbm.py --build --past')


def describe(split):
    uid, y = labels(split)
    per_user = pd.DataFrame({'u': uid, 'y': y}).groupby('u')['y'].agg(['sum', 'count'])
    n_users = len(per_user)
    no_pos = int((per_user['sum'] == 0).sum())
    all_pos = int((per_user['sum'] == per_user['count']).sum())
    perfect = evaluate(uid.tolist(), y.tolist(), y.tolist())
    rng_scores = []
    for seed in range(5):
        s = np.random.default_rng(seed).random(len(y))
        rng_scores.append(evaluate(uid.tolist(), y.tolist(), s.tolist())['primary'])
    return {
        'rows': int(len(y)), 'users': n_users, 'positive_rate': round(float(y.mean()), 4),
        'users_with_no_positive': no_pos, 'share_no_positive': round(no_pos / n_users, 4),
        'users_all_positive': all_pos, 'share_all_positive': round(all_pos / n_users, 4),
        'label_perfect': {k: round(v, 4) for k, v in perfect.items() if isinstance(v, float)},
        'random_primary_mean_of_5_seeds': round(float(np.mean(rng_scores)), 4),
    }


def main():
    doc = {'note': 'label-perfect = score each impression by its own label; random = uniform noise, mean of 5 seeds; '
                   'official evaluate.py, unmodified',
           'valid': describe('valid'), 'test': describe('test')}
    t = doc['test']
    lo, hi = t['random_primary_mean_of_5_seeds'], t['label_perfect']['primary']
    doc['test']['position_in_range'] = {
        'fm_published_0.5946': round((0.5946 - lo) / (hi - lo), 4),
        'final_0.6015': round((0.6015 - lo) / (hi - lo), 4),
    }
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=2)
        fh.write('\n')
    print(json.dumps(doc, indent=2))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
