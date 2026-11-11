"""Compute the attainable range of the official metric on this benchmark.

    python scripts/official_ceiling.py           # writes results/official/ceiling.json
    python scripts/official_ceiling.py --check   # recompute and compare with the committed file; exit 1 on drift

"Label-perfect" scores every impression by its own label — the best any
ranker can do — and "random" scores with uniform noise (mean of 5 seeds). The
README quotes these to put +0.0069 on a scale: the ceiling is well below 1.0
because users with no positive label contribute nDCG = 0 by protocol and users
with all-positive labels are excluded from GAUC.

Labels come from the feature cache when it has been built, otherwise from the
committed results/official/labels_valid_test.npz (digest-guarded by
official_labels.py), so --check runs on a bare checkout. The two reference
points placed on the range are the published FM test score and the
submission's test primary from final.json, both rounded to 4 decimals as the
README quotes them; the rounded score is also part of the key name, so --check
fails if final.json and ceiling.json stop agreeing on it.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate  # noqa: E402
import official_labels  # noqa: E402  (committed labels + their digest guard)

OUT = os.path.join(ROOT, 'results', 'official', 'ceiling.json')
FINAL_JSON = os.path.join(ROOT, 'results', 'official', 'final.json')
CACHES = [os.path.join(ROOT, 'data', 'interim', 'official_feats_past'),
          os.path.join(ROOT, 'data', 'interim', 'official_feats')]


def labels(split):
    for cache in CACHES:
        p = os.path.join(cache, f'{split}.parquet')
        if os.path.exists(p):
            df = pd.read_parquet(p, columns=['user_id', 'long_view'])
            return df['user_id'].to_numpy(), df['long_view'].to_numpy().astype(np.float64)
    uid, y = official_labels.split_labels(split)
    if uid is None:
        raise SystemExit('no labels: neither the feature cache nor '
                         'results/official/labels_valid_test.npz is present')
    return uid, y.astype(np.float64)


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


def reference_points():
    """(key, score) for the two systems placed on the range, read from final.json."""
    with open(FINAL_JSON, encoding='utf-8') as fh:
        final = json.load(fh)
    fm = round(final['baseline']['fm_published']['test'], 4)
    ours = round(final['metrics']['test']['primary'], 4)
    return [(f'fm_published_{fm:.4f}', fm), (f'final_{ours:.4f}', ours)]


def compute():
    doc = {'note': 'label-perfect = score each impression by its own label; random = uniform noise, mean of 5 seeds; '
                   'official evaluate.py, unmodified',
           'valid': describe('valid'), 'test': describe('test')}
    t = doc['test']
    lo, hi = t['random_primary_mean_of_5_seeds'], t['label_perfect']['primary']
    doc['test']['position_in_range'] = {
        key: round((score - lo) / (hi - lo), 4) for key, score in reference_points()}
    return doc


def diff(want, got, path=''):
    """Every leaf where the committed document and the recomputation differ."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = []
        for k in sorted(set(want) | set(got)):
            if k not in got:
                out.append(f'{path}{k}: in ceiling.json but not recomputed')
            elif k not in want:
                out.append(f'{path}{k}: recomputed but missing from ceiling.json')
            else:
                out.extend(diff(want[k], got[k], f'{path}{k}.'))
        return out
    return [] if want == got else [f'{path.rstrip(".")}: ceiling.json {want!r} vs recomputed {got!r}']


def check():
    with open(OUT, encoding='utf-8') as fh:
        committed = json.load(fh)
    problems = diff(committed, compute())
    for p in problems:
        print('MISMATCH:', p)
    if not problems:
        t = committed['test']
        print(f"ceiling.json reproduced: test random {t['random_primary_mean_of_5_seeds']} -> "
              f"label-perfect {t['label_perfect']['primary']}; "
              f"position_in_range {t['position_in_range']}")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='verify instead of rewriting')
    if ap.parse_args().check:
        sys.exit(check())
    doc = compute()
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(doc, fh, indent=2)
        fh.write('\n')
    print(json.dumps(doc, indent=2))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
