"""Equal-weight rank-average a group of runs into one pseudo-model.

python scripts/official_aggregate.py --name gbdt5 --members a,b,c [--splits valid,test]
Writes results/official/preds/<name>_<split>.npy.
"""
import argparse, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from official_blend import load_preds, rank01
from official_lgbm import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--members', required=True)
    ap.add_argument('--splits', default='valid,test')
    a = ap.parse_args()
    members = a.members.split(',')
    for split in a.splits.split(','):
        agg = sum(rank01(load_preds(m, split)) for m in members) / len(members)
        np.save(os.path.join(OUT, 'preds', f'{a.name}_{split}.npy'), agg)
        print(f"{a.name}_{split}: {len(members)} members aggregated")


if __name__ == '__main__':
    main()
