"""The `long_view` labels of the two scored splits, so the headline score can
be recomputed without the 47 MB dataset download.

    python scripts/official_labels.py           # rebuild the npz from a local source
    python scripts/official_labels.py --check   # re-derive from a local source and compare

`results/official/labels_valid_test.npz` holds two arrays per split — the
`user_id` grouping key (int32) and the `long_view` label (uint8) — in exactly
the row order of `results/official/preds/*.npy`. That is everything
`kuairand-starter-kit/evaluate.py` needs, so `official_final.py --check` can
score the committed predictions offline and compare the result with
`results/official/final.json`.

The npz is a derived file, not a source of truth. It is reproduced from the
public KuaiRand-Pure release by six lines of pandas, and `--check` runs them:

    concat log_standard_4_08_to_4_21_pure.csv and log_standard_4_22_to_5_08_pure.csv
    keep date in [20220422, 20220428] (valid) / [20220429, 20220508] (test)
    stable-sort by user_id, reset the index

which is the same slice-and-sort `official_lgbm.py --build` applies before it
writes the feature cache (see its `SPLITS` and the stable sort at the end of
`build()`). Either source works: the raw CSVs (md5-pinned by
`scripts/download_kuairand.py`) or an already-built feature cache.

`EXPECTED` records the row count and the SHA-256 of each array's raw bytes, so
a tampered npz fails `load()` even where no source is available to re-derive
from — and faking a score would have to edit this file too.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, 'results', 'official', 'labels_valid_test.npz')
DATA_DIR = os.path.join(ROOT, 'data', 'raw', 'KuaiRand-Pure', 'data')
LOGS = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')
CACHES = [os.path.join(ROOT, 'data', 'interim', 'official_feats_past'),
          os.path.join(ROOT, 'data', 'interim', 'official_feats')]

# Must match official_lgbm.SPLITS.
SPLITS = {'valid': (20220422, 20220428), 'test': (20220429, 20220508)}

EXPECTED = {
    'valid_user_id': ('int32', 124909, '5b87bf31a3af0cc39dd8161bc7ee5ee5079223a6fcc80ad3238568752df782e1'),
    'valid_long_view': ('uint8', 124909, '35abd7c20c1f2890ed33b6b8004b55cd81d7f9a0533d7be6fb69bc5bc6b807dd'),
    'test_user_id': ('int32', 170588, 'c9973a83e226a1e6c723437e7b9346f20aa292a8b2eb109c6007e09b0c163fcc'),
    'test_long_view': ('uint8', 170588, '7d67ef2d22a654a22ed36e814a390d909ff28fc168c594ee0e3b7418c477352a'),
}


def rel(path):
    """`path` relative to the repository for messages. os.path.relpath raises
    ValueError on Windows when the two paths are on different drives (a pytest
    tmp_path on C: and a checkout on D:, as on GitHub's Windows runners), which
    would turn a refusal into a crash, so fall back to the absolute path."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path)


def digest(a):
    """SHA-256 of the array's raw little-endian bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _pack(uid, lv):
    return np.asarray(uid).astype(np.int32), np.asarray(lv).astype(np.uint8)


def derive_from_raw():
    """Slice the two public standard logs the way official_lgbm.py does."""
    import pandas as pd
    frames = [pd.read_csv(os.path.join(DATA_DIR, f),
                          usecols=['user_id', 'date', 'long_view']) for f in LOGS]
    df = pd.concat(frames, ignore_index=True)
    out = {}
    for split, (lo, hi) in SPLITS.items():
        part = (df[(df['date'] >= lo) & (df['date'] <= hi)]
                .sort_values('user_id', kind='stable').reset_index(drop=True))
        out[f'{split}_user_id'], out[f'{split}_long_view'] = _pack(
            part['user_id'].to_numpy(), part['long_view'].to_numpy())
    return out


def derive_from_cache(cache):
    """Same two columns, read back out of a built feature cache."""
    import pandas as pd
    out = {}
    for split in SPLITS:
        df = pd.read_parquet(os.path.join(cache, f'{split}.parquet'),
                             columns=['user_id', 'long_view'])
        out[f'{split}_user_id'], out[f'{split}_long_view'] = _pack(
            df['user_id'].to_numpy(), df['long_view'].to_numpy())
    return out


def source():
    """(arrays, description) from whichever local source exists, else (None, why)."""
    if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in LOGS):
        return derive_from_raw(), f'the raw logs in {rel(DATA_DIR)}'
    for cache in CACHES:
        if all(os.path.exists(os.path.join(cache, f'{s}.parquet')) for s in SPLITS):
            return derive_from_cache(cache), f'the feature cache {rel(cache)}'
    return None, ('no local source: run scripts/download_kuairand.py (or '
                  'scripts/official_lgbm.py --build --past)')


def load():
    """The committed labels, refusing anything that does not match EXPECTED."""
    if not os.path.exists(NPZ):
        return None
    with np.load(NPZ) as z:
        arrays = {k: z[k] for k in EXPECTED}
    for key, (dtype, rows, sha) in EXPECTED.items():
        a = arrays[key]
        if a.dtype != np.dtype(dtype) or len(a) != rows or digest(a) != sha:
            raise SystemExit(
                f'{rel(NPZ)}: {key} does not match the digest '
                f'recorded in scripts/official_labels.py - refusing to score '
                f'against it. Rebuild it with scripts/official_labels.py.')
    return arrays


def split_labels(split):
    """(user_id, long_view) for one split, or (None, None) if the npz is absent."""
    arrays = load()
    if arrays is None:
        return None, None
    return arrays[f'{split}_user_id'], arrays[f'{split}_long_view']


def check():
    arrays, where = source()
    if arrays is None:
        print(f'cannot verify: {where}')
        return 1
    problems = []
    for key, (dtype, rows, sha) in EXPECTED.items():
        a = arrays.get(key)
        if a is None:
            problems.append(f'{key}: missing from {where}')
            continue
        if len(a) != rows:
            problems.append(f'{key}: {len(a)} rows re-derived vs {rows} expected')
        elif digest(a) != sha:
            problems.append(f'{key}: re-derived bytes differ from the recorded sha256')
    committed = None
    if os.path.exists(NPZ):
        with np.load(NPZ) as z:
            committed = {k: z[k] for k in z.files}
        for key in EXPECTED:
            if key not in committed:
                problems.append(f'{key}: missing from the committed npz')
            elif not np.array_equal(committed[key], arrays[key]):
                problems.append(f'{key}: committed npz differs from {where}')
    else:
        problems.append(f'{rel(NPZ)} is missing')
    for p in problems:
        print('MISMATCH:', p)
    if problems:
        return 1
    print(f're-derived from {where}')
    for key, (_, rows, sha) in EXPECTED.items():
        print(f'  {key}: {rows} rows, sha256 {sha[:16]} - identical to the committed npz')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='re-derive from a local source and compare instead of rewriting')
    a = ap.parse_args()
    if a.check:
        return check()
    arrays, where = source()
    if arrays is None:
        print(where)
        return 1
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez_compressed(NPZ, **arrays)
    print(f'wrote {NPZ} from {where}')
    for key in sorted(arrays):
        print(f"    '{key}': ('{arrays[key].dtype}', {len(arrays[key])}, "
              f"'{digest(arrays[key])}'),")
    return 0


if __name__ == '__main__':
    sys.exit(main())
