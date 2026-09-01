"""Torch FM on long_view (official split) -> per-row score features for GBDT.

Fields: user, video, author, tab, dur_bucket (same as starter-kit FM).
Full model early-stops on valid primary; the best epoch count is reused to
train 5 fold models that produce out-of-fold train scores (no self-leakage).

Outputs results/official/fm_feats/{train,valid,test}.npy aligned to the
feature-cache row order, plus fm_meta.json with the full model's valid metric.
"""
import json, os, sys, time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate
from official_lgbm import CACHE, OUT, OfficialFeval

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--k', type=int, default=32)
_ap.add_argument('--seed', type=int, default=0)
_ap.add_argument('--suffix', default='', help='output dir fm_feats<suffix>')
_ap.add_argument('--no_oof', action='store_true',
                 help='skip fold models (blend-only variant)')
_args = _ap.parse_args()

FEAT_DIR = os.path.join(OUT, 'fm_feats' + _args.suffix)
K = _args.k
LR = 1e-3
BS = 32768
MAX_EPOCHS = 40
PATIENCE = 4
SEED = _args.seed
FOLDS = 5


class FM(torch.nn.Module):
    def __init__(self, dim, k=K):
        super().__init__()
        self.emb = torch.nn.Embedding(dim, k)
        self.lin = torch.nn.Embedding(dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, 0, 0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)                       # (B,F,k)
        s = e.sum(1)
        inter = 0.5 * ((s * s).sum(1) - (e * e).sum((1, 2)))
        return self.bias + self.lin(x).sum((1, 2)) + inter


def encode_all():
    parts = {n: pd.read_parquet(os.path.join(CACHE, f'{n}.parquet'),
                                columns=['user_id', 'video_id', 'author_id_c',
                                         'tab', 'duration_ms', 'long_view'])
             for n in ('train', 'valid', 'test')}
    tr = parts['train']
    edges = np.quantile(tr['duration_ms'].to_numpy(), np.linspace(0, 1, 11)[1:-1])

    def raw_cols(df):
        return [df['user_id'].to_numpy(),
                df['video_id'].to_numpy(),
                df['author_id_c'].astype('int64').to_numpy(),
                df['tab'].astype('int64').to_numpy(),
                np.searchsorted(edges, df['duration_ms'].to_numpy())]

    vocabs, X = [], {}
    tr_cols = raw_cols(tr)
    for col in tr_cols:
        u = pd.unique(col)
        vocabs.append({v: i for i, v in enumerate(u)})
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1])
    for name, df in parts.items():
        cols = raw_cols(df)
        enc = np.empty((len(df), len(cols)), dtype=np.int64)
        for i, col in enumerate(cols):
            m = vocabs[i]
            unk = len(m)
            enc[:, i] = np.fromiter((m.get(v, unk) for v in col),
                                    dtype=np.int64, count=len(col)) + offsets[i]
        X[name] = enc
    y = {n: parts[n]['long_view'].to_numpy().astype(np.float32)
         for n in parts}
    users = {n: parts[n]['user_id'].to_numpy() for n in parts}
    return X, y, users, int(sum(dims))


def train_one(Xtr, ytr, dim, epochs, Xva=None, feval=None, yva=None, seed=SEED):
    torch.manual_seed(seed)
    m = FM(dim)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)
    rng = np.random.default_rng(seed)
    best, bad, best_state, best_ep = -1.0, 0, None, epochs
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        m.train()
        tot = 0.0
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            opt.zero_grad()
            loss = lossf(m(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if Xva is not None:
            m.eval()
            with torch.no_grad():
                p = predict(m, Xva)
            pr = feval.primary(yva, p)
            print(f"  ep {ep:2d} loss {tot/len(ytr):.4f} valid~primary {pr:.4f} "
                  f"{time.time()-t0:.1f}s", flush=True)
            if pr > best + 1e-5:
                best, bad, best_ep = pr, 0, ep
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, best_ep


def predict(m, X, bs=200_000):
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(torch.from_numpy(X[i:i + bs])).numpy())
    return np.concatenate(out)


def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    t0 = time.time()
    X, y, users, dim = encode_all()
    print(f"encoded dim={dim} {time.time()-t0:.1f}s", flush=True)
    feval = OfficialFeval(users['valid'])

    print("full model (early stop on valid primary):", flush=True)
    full, best_ep = train_one(X['train'], y['train'], dim, MAX_EPOCHS,
                              Xva=X['valid'], feval=feval, yva=y['valid'])
    for name in ('valid', 'test'):
        np.save(os.path.join(FEAT_DIR, f'{name}.npy'), predict(full, X[name]))
    r = evaluate(users['valid'].tolist(), y['valid'].tolist(),
                 predict(full, X['valid']).tolist())
    print(f"full FM valid: GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
          f"primary {r['primary']:.4f} (best_ep {best_ep})", flush=True)

    if _args.no_oof:
        meta = {'k': K, 'lr': LR, 'seed': SEED, 'best_ep': best_ep,
                'full_valid': {k: round(v, 6) if isinstance(v, float) else v
                               for k, v in r.items()},
                'wall_seconds': round(time.time() - t0, 1)}
        with open(os.path.join(FEAT_DIR, 'fm_meta.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)
        print(f"done (no oof) in {meta['wall_seconds']}s", flush=True)
        return

    oof = np.zeros(len(y['train']), dtype=np.float32)
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, FOLDS, len(y['train']))
    for f in range(FOLDS):
        tr_idx = np.where(fold != f)[0]
        te_idx = np.where(fold == f)[0]
        print(f"fold {f}: train {len(tr_idx)} predict {len(te_idx)}", flush=True)
        mf, _ = train_one(X['train'][tr_idx], y['train'][tr_idx], dim,
                          best_ep, seed=SEED + 100 + f)
        oof[te_idx] = predict(mf, X['train'][te_idx])
    np.save(os.path.join(FEAT_DIR, 'train.npy'), oof)
    meta = {'k': K, 'lr': LR, 'best_ep': best_ep, 'folds': FOLDS,
            'full_valid': {k: round(v, 6) if isinstance(v, float) else v
                           for k, v in r.items()},
            'wall_seconds': round(time.time() - t0, 1)}
    with open(os.path.join(FEAT_DIR, 'fm_meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"done in {meta['wall_seconds']}s", flush=True)


if __name__ == '__main__':
    main()
