"""Quantify how much feature staleness costs, using train+valid data only.

Both caches use a 7-day aggregate window, so window LENGTH is held constant
and only the GAP to the scored window differs:
  A: stats 04/15-04/21 -> gap 0 days to valid (04/22-)
  B: stats 04/08-04/14 -> gap 7 days to valid
A - B isolates the value of a 7-day-fresher statistics pool. Test rows in the
default pipeline sit 7-14 days after the train window, so this estimates what
fresher statistics would be worth there. No test data is touched.
"""
import os, sys

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate
from official_lgbm import CACHE, OUT, GROUPS, CATS, SIDECAR

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'p4_ease'
FEATS = ('base,item,user,author,ua,uv,utag,vside,vstat,uside,ud,utab,utags,'
         'itemcf,fm,sess,ease')


def score(cache_suffix, cols, cats, booster, label):
    p = pd.read_parquet(os.path.join(CACHE + cache_suffix, 'valid.parquet'),
                        columns=['user_id', 'long_view']
                        + [c for c in cols if c not in {'fm_score'}])
    p['fm_score'] = np.load(os.path.join(OUT, 'fm_feats', 'valid.npy')
                            ).astype('float32')
    for c in cats:
        p[c] = p[c].astype('category')
    pred = booster.predict(p[cols])
    r = evaluate(p['user_id'].tolist(), p['long_view'].tolist(), pred.tolist())
    print(f"{label:34s} GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
          f"primary {r['primary']:.4f}")
    return r['primary']


def main():
    cols = [c for g in FEATS.split(',') for c in GROUPS[g]]
    cats = [c for c in cols if c in CATS]
    booster = lgb.Booster(model_str=open(
        os.path.join(OUT, f'{MODEL}.txt')).read())
    print(f"model {MODEL}, scoring the same valid rows with different "
          f"statistics pools:\n")
    a = score('_staleA', cols, cats, booster, 'A: 7d stats, gap 0 (fresh)')
    b = score('_staleB', cols, cats, booster, 'B: 7d stats, gap 7 (stale)')
    full = score('_past', cols, cats, booster, 'reference: 14d stats, gap 0')
    print(f"\nstaleness cost of a 7-day gap (A - B): {a - b:+.4f} primary")
    print(f"7d-fresh vs 14d-full (A - full):       {a - full:+.4f} primary")


if __name__ == '__main__':
    main()
