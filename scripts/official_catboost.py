"""CatBoost member on the same past-only feature cache (library diversity).

CatBoost's ordered target statistics handle the high-cardinality categoricals
differently from LightGBM's split-based handling, so it is a genuinely
different model class for the blend, not another seed.

python scripts/official_catboost.py --name cb1 [--iters 2000] [--seed 0]
Writes results/official/preds/<name>_{valid,test}.npy and <name>.json.
"""
import argparse, json, os, sys, time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate
from official_lgbm import CACHE, OUT, GROUPS, CATS, SIDECAR, OfficialFeval

FEATS = ('base,item,user,author,ua,uv,utag,vside,vstat,uside,ud,utab,utags,'
         'itemcf,fm,sess,ease')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cb1')
    ap.add_argument('--features', default=FEATS)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--l2', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    groups = a.features.split(',')
    cols = [c for g in groups for c in GROUPS[g]]
    sidecars = [SIDECAR[g] for g in groups if g in SIDECAR]
    side_cols = {c for _, c in sidecars}
    pq_cols = [c for c in cols if c not in side_cols]
    cats = [c for c in cols if c in CATS]

    parts = {}
    for name in ('train', 'valid', 'test'):
        p = pd.read_parquet(os.path.join(CACHE + '_past', f'{name}.parquet'),
                            columns=['user_id', 'long_view'] + pq_cols)
        for subdir, col in sidecars:
            p[col] = np.load(os.path.join(OUT, subdir, f'{name}.npy')
                             ).astype('float32')
        for c in cats:      # CatBoost takes categoricals as strings, no NaN
            p[c] = p[c].astype('object').where(p[c].notna(), 'NA').astype(str)
        parts[name] = p
    print(f"loaded {time.time()-t0:.0f}s, {len(cols)} features, "
          f"{len(cats)} categorical", flush=True)

    dtr = Pool(parts['train'][cols], parts['train']['long_view'],
               cat_features=cats)
    dva = Pool(parts['valid'][cols], parts['valid']['long_view'],
               cat_features=cats)
    m = CatBoostClassifier(
        iterations=a.iters, learning_rate=a.lr, depth=a.depth,
        l2_leaf_reg=a.l2, random_seed=a.seed, eval_metric='AUC',
        od_type='Iter', od_wait=100, thread_count=8, verbose=200,
        bootstrap_type='Bernoulli', subsample=0.8)
    m.fit(dtr, eval_set=dva, use_best_model=True)

    res = {'name': a.name, 'features': a.features, 'library': 'catboost',
           'best_iter': int(m.get_best_iteration()),
           'params': {'lr': a.lr, 'depth': a.depth, 'l2': a.l2,
                      'seed': a.seed}}
    for name in ('valid', 'test'):
        p = m.predict_proba(parts[name][cols])[:, 1]
        np.save(os.path.join(OUT, 'preds', f'{a.name}_{name}.npy'), p)
        if name == 'valid':
            r = evaluate(parts[name]['user_id'].tolist(),
                         parts[name]['long_view'].tolist(), p.tolist())
            res['valid'] = {k: round(v, 6) if isinstance(v, float) else v
                            for k, v in r.items()}
    res['wall_seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, f'{a.name}.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    v = res['valid']
    print(f"{a.name}: valid GAUC {v['GAUC']:.4f} nDCG@5 {v['nDCG@5']:.4f} "
          f"primary {v['primary']:.4f} | iter {res['best_iter']} | "
          f"{res['wall_seconds']}s (test preds saved, unscored)", flush=True)


if __name__ == '__main__':
    main()
