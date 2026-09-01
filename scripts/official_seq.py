"""Tiny DIN-style target-attention sequence model as a blend member (CPU).

History for every impression = the user's train-window long_view events with
time_ms strictly earlier than the impression (last 50). Train rows therefore
use only their own past; valid/test rows use only train-window history —
no evaluation-window label ever enters a feature (FAQ 2.9.3).

Outputs results/official/preds/seq_{valid,test}.npy aligned to the feature
cache row order. Prints the valid metric only; test is predicted blind.
"""
import os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(6)
torch.manual_seed(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'kuairand-starter-kit'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import evaluate
from official_lgbm import DATA_DIR, OUT, SPLITS, OfficialFeval

N_HIST = 50
K = 32
BS = 4096
MAX_EPOCHS = 4
LR = 1e-3


def load_splits():
    """Reproduce the feature-cache row order exactly (concat -> date filter ->
    reset_index -> stable sort by user_id)."""
    logs = []
    for f in ('log_standard_4_08_to_4_21_pure.csv',
              'log_standard_4_22_to_5_08_pure.csv'):
        logs.append(pd.read_csv(os.path.join(DATA_DIR, f), usecols=[
            'user_id', 'video_id', 'date', 'time_ms', 'long_view',
            'duration_ms', 'tab']))
    df = pd.concat(logs, ignore_index=True)
    parts = {}
    for name, (lo, hi) in SPLITS.items():
        p = df[(df['date'] >= lo) & (df['date'] <= hi)].copy().reset_index(drop=True)
        parts[name] = p.sort_values('user_id', kind='stable').reset_index(drop=True)
    return parts


def build_arrays(parts):
    tr = parts['train']
    vids = pd.unique(tr['video_id'])
    v2c = {v: i + 1 for i, v in enumerate(vids)}          # 0 = pad
    unk = len(vids) + 1
    n_items = len(vids) + 2
    edges = np.quantile(tr['duration_ms'].to_numpy(), np.linspace(0, 1, 11)[1:-1])

    lv = tr[tr['long_view'] == 1][['user_id', 'time_ms', 'video_id']]
    lv = lv.sort_values(['user_id', 'time_ms'], kind='stable')
    hist_t = {u: g['time_ms'].to_numpy(dtype='int64')
              for u, g in lv.groupby('user_id')}
    hist_v = {u: np.fromiter((v2c[v] for v in g['video_id']), dtype='int32',
                             count=len(g))
              for u, g in lv.groupby('user_id')}

    out = {}
    for name, p in parts.items():
        n = len(p)
        H = np.zeros((n, N_HIST), dtype='int32')
        L = np.zeros(n, dtype='int32')
        pu = p['user_id'].to_numpy()
        pt = p['time_ms'].to_numpy(dtype='int64')
        for u, rows in p.groupby('user_id').indices.items():
            ht = hist_t.get(u)
            if ht is None:
                continue
            hv = hist_v[u]
            pos = np.searchsorted(ht, pt[rows], side='left')
            for r, e in zip(rows, pos):
                s = max(0, e - N_HIST)
                if e > s:
                    H[r, :e - s] = hv[s:e][::-1]          # most recent first
                    L[r] = e - s
        cand = np.fromiter((v2c.get(v, unk) for v in p['video_id']),
                           dtype='int32', count=n)
        dur = np.searchsorted(edges, p['duration_ms'].to_numpy()).astype('int32')
        tab = p['tab'].to_numpy().astype('int32')
        y = p['long_view'].to_numpy().astype('float32')
        out[name] = dict(H=H, L=L, cand=cand, dur=dur, tab=tab, y=y,
                         users=p['user_id'].to_numpy())
        print(f"{name}: rows {n}, mean hist len {L.mean():.1f}", flush=True)
    return out, n_items


class TinyDIN(nn.Module):
    def __init__(self, n_items, k=K):
        super().__init__()
        self.item = nn.Embedding(n_items, k, padding_idx=0)
        self.dur = nn.Embedding(11, 8)
        self.tab = nn.Embedding(16, 8)
        self.att = nn.Sequential(nn.Linear(3 * k, 32), nn.ReLU(),
                                 nn.Linear(32, 1))
        self.top = nn.Sequential(nn.Linear(3 * k + 16, 64), nn.ReLU(),
                                 nn.Linear(64, 1))
        nn.init.normal_(self.item.weight, 0, 0.01)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, cand, hist, hlen, dur, tab):
        ec = self.item(cand)                              # B,k
        eh = self.item(hist)                              # B,N,k
        ecx = ec.unsqueeze(1).expand_as(eh)
        w = self.att(torch.cat([ecx, eh, ecx * eh], -1)).squeeze(-1)
        mask = (torch.arange(hist.shape[1])[None, :] < hlen[:, None])
        w = w.masked_fill(~mask, -1e9)
        a = torch.softmax(w, -1)
        pooled = (a.unsqueeze(-1) * eh).sum(1)
        pooled = pooled * (hlen > 0).float().unsqueeze(-1)
        x = torch.cat([ec, pooled, ec * pooled,
                       self.dur(dur), self.tab(tab)], -1)
        return self.top(x).squeeze(-1)


def batches(d, idx, bs=BS):
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        yield (torch.from_numpy(d['cand'][b].astype('int64')),
               torch.from_numpy(d['H'][b].astype('int64')),
               torch.from_numpy(d['L'][b].astype('int64')),
               torch.from_numpy(d['dur'][b].astype('int64')),
               torch.from_numpy(d['tab'][b].astype('int64')),
               torch.from_numpy(d['y'][b]))


def predict(m, d):
    m.eval()
    out = []
    with torch.no_grad():
        for c, h, l, du, tb, _ in batches(d, np.arange(len(d['y'])), 20000):
            out.append(m(c, h, l, du, tb).numpy())
    return np.concatenate(out)


def train_model(data, n_items, epochs, fe=None, yva=None, seed=0, rows=None):
    torch.manual_seed(seed)
    m = TinyDIN(n_items)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    pool = np.arange(len(data['train']['y'])) if rows is None else rows
    best, best_state, best_ep = -1.0, None, epochs
    for ep in range(1, epochs + 1):
        te = time.time()
        m.train()
        tot = 0.0
        for c, h, l, du, tb, y in batches(data['train'], rng.permutation(pool)):
            opt.zero_grad()
            loss = lossf(m(c, h, l, du, tb), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
        if fe is None:
            print(f"  ep {ep} loss {tot/len(pool):.4f} {time.time()-te:.0f}s",
                  flush=True)
            continue
        pr = fe.primary(yva, predict(m, data['valid']))
        print(f"ep {ep} loss {tot/len(pool):.4f} valid~primary {pr:.4f} "
              f"{time.time()-te:.0f}s", flush=True)
        if pr > best + 1e-5:
            best, best_ep = pr, ep
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, best_ep


def main():
    oof = '--oof' in sys.argv
    t0 = time.time()
    parts = load_splits()
    data, n_items = build_arrays(parts)
    print(f"prep {time.time()-t0:.0f}s, items {n_items}", flush=True)
    fe = OfficialFeval(data['valid']['users'])
    yva = data['valid']['y'].astype('float64')
    m, best_ep = train_model(data, n_items, MAX_EPOCHS, fe, yva)
    pva = predict(m, data['valid'])
    r = evaluate(data['valid']['users'].tolist(), yva.tolist(), pva.tolist())
    print(f"seq valid: GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
          f"primary {r['primary']:.4f} (best_ep {best_ep})", flush=True)
    np.save(os.path.join(OUT, 'preds', 'seq_valid.npy'), pva)
    np.save(os.path.join(OUT, 'preds', 'seq_test.npy'), predict(m, data['test']))

    if oof:
        # out-of-fold train scores so the GBDT can consume seq as a feature
        os.makedirs(os.path.join(OUT, 'seq_feats'), exist_ok=True)
        n = len(data['train']['y'])
        fold = np.random.default_rng(0).integers(0, 5, n)
        oofp = np.zeros(n, dtype='float32')
        for f in range(5):
            tr_rows = np.where(fold != f)[0]
            te_rows = np.where(fold == f)[0]
            print(f"fold {f}: train {len(tr_rows)} predict {len(te_rows)}",
                  flush=True)
            mf, _ = train_model(data, n_items, best_ep, seed=100 + f,
                                rows=tr_rows)
            sub = {k: v[te_rows] for k, v in data['train'].items()}
            oofp[te_rows] = predict(mf, sub)
        for name, arr in (('train', oofp), ('valid', pva),
                          ('test', np.load(os.path.join(OUT, 'preds',
                                                        'seq_test.npy')))):
            np.save(os.path.join(OUT, 'seq_feats', f'{name}.npy'), arr)
        print("seq OOF features written", flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
