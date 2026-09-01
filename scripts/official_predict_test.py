"""Predict the test split with an already-trained LightGBM run (no retrain).

python scripts/official_predict_test.py --name p_ff05 [--name p_rank ...]
Writes results/official/preds/<name>_test.npy and prints the official test
metric. Valid/test features are identical between the base and past caches,
so either cache serves; we read the base one.
"""
import argparse, json, os, sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate
from official_lgbm import CACHE, OUT, GROUPS, CATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', action='append', required=True)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    te_full = pd.read_parquet(os.path.join(CACHE + '_past', 'test.parquet'))
    fm_test = None
    for name in a.name:
        with open(os.path.join(OUT, f'{name}.json')) as fh:
            meta = json.load(fh)
        cols = [c for g in meta['features'].split(',') for c in GROUPS[g]]
        te = te_full.copy()
        if 'fm_score' in cols:
            if fm_test is None:
                fm_test = np.load(os.path.join(OUT, 'fm_feats', 'test.npy'))
            te['fm_score'] = fm_test.astype('float32')
        for c in cols:
            if c in CATS:
                te[c] = te[c].astype('category')
        booster = lgb.Booster(model_str=open(
            os.path.join(OUT, f'{name}.txt')).read())
        p = booster.predict(te[cols])
        np.save(os.path.join(OUT, 'preds', f'{name}_test.npy'), p)
        if not a.quiet:
            r = evaluate(te['user_id'].tolist(), te['long_view'].tolist(),
                         p.tolist())
            print(f"{name}: test GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
                  f"primary {r['primary']:.4f}")
        else:
            print(f"{name}: test preds saved ({len(p)} rows)")


if __name__ == '__main__':
    main()
