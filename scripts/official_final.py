"""Reproduce the designated submission from the committed member predictions,
score it with the official evaluator, and write results/official/final.json.

    python scripts/official_final.py            # rebuild FINAL_{valid,test}.npy and final.json
    python scripts/official_final.py --check    # recompute and compare with the committed artifacts; exit 1 on drift

The recipe is the one recorded in final.json:

    FINAL = 0.80 * rank01(gbdt5v2) + 0.20 * rank01(rank3)

where gbdt5v2 and rank3 are the equal-weight rank averages that
official_aggregate.py wrote from their members (5 LightGBM seeds; 3 lambdarank
variants). Only the two aggregate prediction files per split are needed, so
the headline number can be recomputed without retraining anything.

Scoring needs the labels. They come from the feature cache when it has been
built (scripts/download_kuairand.py, then official_lgbm.py --build --past),
and otherwise from results/official/labels_valid_test.npz, the 140 KB
(user_id, long_view) slice that scripts/official_labels.py derives from the
same public download. So --check recomputes GAUC / nDCG@5 / primary and holds
them against final.json to 5e-7 on a bare checkout, with no dataset present.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate  # noqa: E402  (official scorer, unmodified)
import official_labels  # noqa: E402  (committed labels + their digest guard)

OUT = os.path.join(ROOT, 'results', 'official')
PREDS = os.path.join(OUT, 'preds')
FINAL_JSON = os.path.join(OUT, 'final.json')
CACHES = [os.path.join(ROOT, 'data', 'interim', 'official_feats_past'),
          os.path.join(ROOT, 'data', 'interim', 'official_feats')]

WEIGHTS = {'gbdt5v2': 0.8, 'rank3': 0.2}
MEMBERS = {
    'gbdt5v2': ['p4_ease', 'p4_ease_s1', 'p4_ease_s2', 'p4_ease_s3', 'p4_ease_s4'],
    'rank3': ['p_rank_t10', 'p3_rank_ff05', 'p3_rank_s1'],
}
PUBLISHED_FM = {'valid': 0.6016, 'test': 0.5946}


def rank01(x):
    """Global rank in [0, 1], ties broken by position (stable) — identical to
    official_blend.rank01, duplicated here so this script imports no
    LightGBM."""
    r = np.argsort(np.argsort(x, kind='stable'), kind='stable')
    return (r / (len(r) - 1)).astype(np.float64)


def blend(group_preds, weights=WEIGHTS):
    """Weighted sum of rank01-transformed group predictions."""
    total = sum(weights.values())
    return sum(weights[g] * rank01(group_preds[g]) for g in weights) / total


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_group(split):
    return {g: np.load(os.path.join(PREDS, f'{g}_{split}.npy')) for g in WEIGHTS}


def labels(split):
    """Ground truth for one split: the feature cache when it has been built,
    otherwise the committed npz — official_labels.load() refuses to hand it
    over unless its bytes match the digests recorded in that script."""
    for cache in CACHES:
        p = os.path.join(cache, f'{split}.parquet')
        if os.path.exists(p):
            import pandas as pd
            df = pd.read_parquet(p, columns=['user_id', 'long_view'])
            return df['user_id'].to_numpy(), df['long_view'].to_numpy().astype(np.float64)
    uid, y = official_labels.split_labels(split)
    if uid is None:
        return None, None
    return uid, y.astype(np.float64)


def score(split, preds):
    uid, y = labels(split)
    if uid is None:
        return None
    if len(uid) != len(preds):
        # A shifted or truncated label array scores to a plausible-looking
        # wrong number instead of raising, so refuse the pair outright.
        raise SystemExit(f'{split}: {len(uid)} labels vs {len(preds)} predictions '
                         f'- refusing to score a misaligned pair')
    m = evaluate(uid.tolist(), y.tolist(), preds.tolist())
    return {'GAUC': round(m['GAUC'], 6), 'nDCG@5': round(m['nDCG@5'], 6),
            'primary': round(m['primary'], 6), 'users': int(m.get('users', len(set(uid.tolist())))),
            'rows': int(m.get('rows', len(uid)))}


def build(write=True):
    result = {'splits': {}, 'artifacts': {}}
    for split in ('valid', 'test'):
        group = load_group(split)
        final = blend(group)
        path = os.path.join(PREDS, f'FINAL_{split}.npy')
        if write:
            np.save(path, final)
        result['splits'][split] = {'preds': final, 'metrics': score(split, final)}
        for g in WEIGHTS:
            result['artifacts'][f'{g}_{split}.npy'] = sha256(os.path.join(PREDS, f'{g}_{split}.npy'))
        if os.path.exists(path):
            result['artifacts'][f'FINAL_{split}.npy'] = sha256(path)
    return result


def final_json(result):
    doc = {
        'designation': 'validation-best checkpoint at run stop (FAQ 2.9.1c)',
        'recipe': {
            'weights': WEIGHTS,
            'gbdt5v2_members': MEMBERS['gbdt5v2'],
            'rank3_members': MEMBERS['rank3'],
            'combine': 'rank01-transform each member, equal-weight within group, then weighted sum',
            'reproduce': 'python scripts/official_final.py',
        },
        'baseline': {'fm_published': PUBLISHED_FM},
        'metrics': {s: result['splits'][s]['metrics'] for s in ('valid', 'test')},
        'artifacts_sha256': result['artifacts'],
    }
    test = doc['metrics']['test']
    if test:
        doc['delta_test_primary_vs_published_fm'] = round(test['primary'] - PUBLISHED_FM['test'], 6)
    return doc


def check():
    problems = []
    committed = json.load(open(FINAL_JSON, encoding='utf-8')) if os.path.exists(FINAL_JSON) else None
    result = build(write=False)
    for split in ('valid', 'test'):
        path = os.path.join(PREDS, f'FINAL_{split}.npy')
        if os.path.exists(path):
            stored = np.load(path)
            if not np.allclose(stored, result['splits'][split]['preds'], atol=1e-12):
                problems.append(f'FINAL_{split}.npy does not equal the recipe applied to the committed group predictions')
            else:
                print(f'FINAL_{split}.npy: reproduced bit for bit from gbdt5v2 + rank3')
        else:
            problems.append(f'FINAL_{split}.npy is missing')
        m = result['splits'][split]['metrics']
        if m is None:
            # The labels ship with the repository, so "cannot score" means an
            # artifact is missing, not that this checkout is unequipped. Fail
            # rather than let --check quietly degrade to the blend-only check.
            problems.append(f'{split}: no labels to score with — '
                            f'results/official/labels_valid_test.npz is missing '
                            f'(rebuild it with scripts/official_labels.py)')
            continue
        print(f"{split}: GAUC {m['GAUC']:.5f}  nDCG@5 {m['nDCG@5']:.5f}  primary {m['primary']:.4f}")
        if committed:
            for key in ('GAUC', 'nDCG@5', 'primary'):
                want = committed['metrics'][split][key]
                if abs(want - m[key]) > 5e-7:
                    problems.append(f'{split} {key}: recomputed {m[key]} vs final.json {want}')
    if committed and 'artifacts_sha256' in committed:
        for name, digest in committed['artifacts_sha256'].items():
            if result['artifacts'].get(name) != digest:
                problems.append(f'{name}: sha256 differs from final.json')
    for p in problems:
        print('MISMATCH:', p)
    if not problems:
        print('final.json and FINAL_*.npy are consistent with the committed group predictions')
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='verify instead of rewriting')
    a = ap.parse_args()
    if a.check:
        return check()
    result = build(write=True)
    doc = final_json(result)
    with open(FINAL_JSON, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=2)
        fh.write('\n')
    for split in ('valid', 'test'):
        m = doc['metrics'][split]
        if m:
            print(f"{split}: GAUC {m['GAUC']:.5f}  nDCG@5 {m['nDCG@5']:.5f}  primary {m['primary']:.4f}")
        else:
            print(f'{split}: blended and saved; labels not available, not scored')
    print(f'wrote {FINAL_JSON}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
