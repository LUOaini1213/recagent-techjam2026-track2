"""LightGBM under the OFFICIAL kuairand-starter-kit protocol.

Label long_view, split train 0408-0421 / valid 0422-0428 / test 0429-0508,
metric mean(GAUC, nDCG@5) computed by kuairand-starter-kit/evaluate.py verbatim.

Stage 1 (once):  python scripts/official_lgbm.py --build
Stage 2 (sweep): python scripts/official_lgbm.py --train --name run1 \
                     --features base,item,user,author,ua,uv,utag,vside,vstat,uside \
                     --objective binary --num_leaves 127
Test is only scored with --eval_test (final model, once).

All history stats come from the train window only; train rows use
leave-one-out versions so a row never sees its own label.
"""
import argparse, json, os, sys, time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, 'kuairand-starter-kit')
sys.path.insert(0, KIT)
from evaluate import evaluate  # official metric, do not reimplement

DATA_DIR = os.path.join(ROOT, 'data', 'raw', 'KuaiRand-Pure', 'data')
CACHE = os.path.join(ROOT, 'data', 'interim', 'official_feats')
OUT = os.path.join(ROOT, 'results', 'official')
PRIOR = 20.0

SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}

GROUPS = {
    'base':  ['tab', 'duration_ms', 'log_dur', 'hour', 'dow'],
    'item':  ['v_imp', 'v_lv_rate', 'v_click_rate', 'v_hate_rate',
              'v_play_ratio', 'v_play_ms'],
    'user':  ['u_imp', 'u_lv_rate', 'u_click_rate', 'u_play_ratio',
              'u_lv_dur', 'dur_diff', 'dur_ratio'],
    'author': ['a_imp', 'a_lv_rate'],
    'ua':    ['ua_cnt', 'ua_lv_rate'],
    'uv':    ['uv_cnt', 'uv_lv_cnt'],
    'utag':  ['ut_cnt', 'ut_lv_rate'],
    'vside': ['video_type', 'upload_type', 'visible_status', 'music_type',
              'tag1', 'server_width', 'server_height', 'aspect',
              'video_duration', 'upload_age'],
    'vstat': ['s_show_cnt', 's_play_cnt', 's_play_user_num', 's_play_progress',
              's_complete_rate', 's_longtime_rate', 's_shorttime_rate',
              's_valid_rate', 's_like_rate', 's_comment_rate', 's_share_rate',
              's_collect_rate', 's_download_rate', 's_follow_rate',
              's_reduce_rate', 's_double_click_rate'],
    'uside': ['user_active_degree', 'is_lowactive_period', 'is_live_streamer',
              'is_video_author', 'follow_user_num', 'fans_user_num',
              'friend_user_num', 'register_days'] +
             [f'onehot_feat{i}' for i in range(18)],
    'ids':   ['user_id_c', 'video_id_c', 'author_id_c'],
    'ud':    ['dur_bucket', 'ud_cnt', 'ud_lv_rate'],
    'utab':  ['utab_cnt', 'utab_lv_rate'],
    'utags': ['utags_cnt', 'utags_lv_rate'],
    'itemcf': ['cf_sum', 'cf_mean', 'u_hist_cnt'],
    'fm':    ['fm_score'],
    'sess':  ['gap_prev_ms', 'impr_idx_day', 'sess_pos', 'uv_seen_today',
              'ua_seen_today', 'ut_seen_today', 'u_days_since', 'v_days_since'],
    'ease':  ['ease_sum', 'ease_mean'],
    'ease2': ['ease50_mean', 'ease1000_mean', 'easeclk_mean'],
    'aux':   ['thresh_ms', 'thresh_ratio', 'v_margin', 'u_margin',
              'v_like_rate', 'v_comment_rate', 'v_forward_rate',
              'v_penter_rate', 'v_pstay'],
    'seq':   ['seq_score'],
}
# feature groups whose values live in .npy sidecars, not the parquet cache
SIDECAR = {'fm': ('fm_feats', 'fm_score'), 'seq': ('seq_feats', 'seq_score')}
CATS = {'tab', 'video_type', 'upload_type', 'visible_status', 'music_type',
        'tag1', 'dur_bucket', 'user_active_degree',
        'user_id_c', 'video_id_c', 'author_id_c',
        } | {f'onehot_feat{i}' for i in range(18)}
CROSS_PRIOR = 5.0


def _smooth(pos, cnt, gmean, prior=PRIOR):
    return (pos + prior * gmean) / (cnt + prior)


def _train_sc(tr, keys, ycol, past):
    """(sum, count) of `ycol` backing each TRAIN row: strictly-before-date
    prefix sums when past=True, else leave-one-out over the full window."""
    if past:
        g = (tr.groupby(keys + ['date'])[ycol].agg(['sum', 'count'])
               .reset_index().sort_values(keys + ['date'], kind='stable'))
        g['ps'] = g.groupby(keys)['sum'].cumsum() - g['sum']
        g['pc'] = g.groupby(keys)['count'].cumsum() - g['count']
        m = g.set_index(keys + ['date'])
        idx = pd.MultiIndex.from_arrays([tr[k] for k in keys] + [tr['date']])
        return (m['ps'].reindex(idx).fillna(0.0).to_numpy(),
                m['pc'].reindex(idx).fillna(0.0).to_numpy())
    if len(keys) == 1:
        g = tr.groupby(keys[0])[ycol].agg(['sum', 'count'])
        s = tr[keys[0]].map(g['sum']).to_numpy().astype('float64')
        c = tr[keys[0]].map(g['count']).to_numpy().astype('float64')
    else:
        g = tr.groupby(keys)[ycol].agg(['sum', 'count'])
        idx = pd.MultiIndex.from_arrays([tr[k] for k in keys])
        s = g['sum'].reindex(idx).to_numpy().astype('float64')
        c = g['count'].reindex(idx).to_numpy().astype('float64')
    return s - tr[ycol].to_numpy(), c - 1.0


def _agg_rate(tr, key, ycol, out_col, feats, past, cnt_col=None):
    """Smoothed rate of `ycol` by `key`; train rows never see their own label."""
    g = tr.groupby(key)[ycol].agg(['sum', 'count'])
    gmean = tr[ycol].mean()
    for name, part in feats.items():
        if name == 'train':
            s, c = _train_sc(tr, [key], ycol, past)
        else:
            s = part[key].map(g['sum']).fillna(0.0).to_numpy()
            c = part[key].map(g['count']).fillna(0.0).to_numpy()
        part[out_col] = _smooth(s, c, gmean).astype('float32')
        if cnt_col:
            part[cnt_col] = c.astype('float32')


def _agg_mean(tr, key, vcol, out_col, feats, past):
    """Mean of `vcol` by `key`; train rows never see their own value."""
    g = tr.groupby(key)[vcol].agg(['sum', 'count'])
    gmean = tr[vcol].mean()
    for name, part in feats.items():
        if name == 'train':
            s, c = _train_sc(tr, [key], vcol, past)
        else:
            s = part[key].map(g['sum']).fillna(0.0).to_numpy()
            c = part[key].map(g['count']).fillna(0.0).to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            m = np.where(c > 0, s / np.maximum(c, 1), gmean)
        part[out_col] = m.astype('float32')


def build(past=False, stats=None, only=None, suffix=None):
    """stats: (lo, hi) overriding which dates form the aggregate source pool
    (default = the train split). only: subset of split names to write.
    suffix: cache directory suffix (default '_past' when past else '')."""
    cache_dir = CACHE + (suffix if suffix is not None
                         else ('_past' if past else ''))
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    logs = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        logs.append(pd.read_csv(os.path.join(DATA_DIR, f), usecols=[
            'user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'is_click',
            'is_hate', 'is_like', 'is_comment', 'is_forward',
            'is_profile_enter', 'profile_stay_time', 'long_view',
            'play_time_ms', 'duration_ms', 'tab']))
    df = pd.concat(logs, ignore_index=True)
    df['profile_stay_time'] = df['profile_stay_time'].fillna(0.0)
    print(f"logs loaded {len(df)} rows {time.time()-t0:.1f}s")

    vb = pd.read_csv(os.path.join(DATA_DIR, 'video_features_basic_pure.csv'))
    vs = pd.read_csv(os.path.join(DATA_DIR, 'video_features_statistic_pure.csv'))
    uf = pd.read_csv(os.path.join(DATA_DIR, 'user_features_pure.csv'))

    df = df.merge(vb[['video_id', 'author_id']], on='video_id', how='left')
    df['author_id'] = df['author_id'].fillna(-1).astype('int64')
    df['dayord'] = (pd.to_datetime(df['date'].astype(str))
                    - pd.Timestamp('2022-04-08')).dt.days.astype('int16')
    df['hour'] = (df['hourmin'] // 100).astype('int16')
    df['dow'] = pd.to_datetime(df['date'].astype(str)).dt.dayofweek.astype('int16')
    df['log_dur'] = np.log1p(df['duration_ms']).astype('float32')
    df['play_ratio'] = (df['play_time_ms'] / df['duration_ms'].clip(lower=1)
                        ).clip(upper=5).astype('float32')

    feats = {}
    for name, (lo, hi) in SPLITS.items():
        if only and name not in only:
            continue
        feats[name] = df[(df['date'] >= lo) & (df['date'] <= hi)].copy().reset_index(drop=True)
    if stats:
        slo, shi = stats
        tr = df[(df['date'] >= slo) & (df['date'] <= shi)].copy().reset_index(drop=True)
        print(f"stats pool overridden: {slo}-{shi}, {len(tr)} rows")
    else:
        tr = feats['train']
    print({k: len(v) for k, v in feats.items()})

    # --- train-window aggregates (train rows never see their own label) ---
    _agg_rate(tr, 'video_id', 'long_view', 'v_lv_rate', feats, past, cnt_col='v_imp')
    _agg_rate(tr, 'video_id', 'is_click', 'v_click_rate', feats, past)
    _agg_rate(tr, 'video_id', 'is_hate', 'v_hate_rate', feats, past)
    _agg_mean(tr, 'video_id', 'play_ratio', 'v_play_ratio', feats, past)
    _agg_mean(tr, 'video_id', 'play_time_ms', 'v_play_ms', feats, past)
    _agg_rate(tr, 'user_id', 'long_view', 'u_lv_rate', feats, past, cnt_col='u_imp')
    _agg_rate(tr, 'user_id', 'is_click', 'u_click_rate', feats, past)
    _agg_mean(tr, 'user_id', 'play_ratio', 'u_play_ratio', feats, past)
    _agg_rate(tr, 'author_id', 'long_view', 'a_lv_rate', feats, past, cnt_col='a_imp')

    # label-mechanism features: long_view is (play_time >= min(duration, 18s))
    # to ~99.8%, so the implied threshold and past margins over it are
    # load-bearing. Row-level play_time is the label's own signal and is
    # NEVER used as a same-row feature — only past-only aggregates of it.
    tr['margin_ms'] = (tr['play_time_ms']
                       - np.minimum(tr['duration_ms'], 18000)).astype('float64')
    for name, part in feats.items():
        th = np.minimum(part['duration_ms'].to_numpy(), 18000)
        part['thresh_ms'] = th.astype('float32')
        part['thresh_ratio'] = (th / np.maximum(
            part['duration_ms'].to_numpy(), 1)).astype('float32')
    _agg_mean(tr, 'video_id', 'margin_ms', 'v_margin', feats, past)
    _agg_mean(tr, 'user_id', 'margin_ms', 'u_margin', feats, past)
    _agg_rate(tr, 'video_id', 'is_like', 'v_like_rate', feats, past)
    _agg_rate(tr, 'video_id', 'is_comment', 'v_comment_rate', feats, past)
    _agg_rate(tr, 'video_id', 'is_forward', 'v_forward_rate', feats, past)
    _agg_rate(tr, 'video_id', 'is_profile_enter', 'v_penter_rate', feats, past)
    _agg_mean(tr, 'video_id', 'profile_stay_time', 'v_pstay', feats, past)

    # mean duration of videos the user long_viewed
    tr['dur_lv'] = (tr['duration_ms'] * tr['long_view']).astype('float64')
    lv_tr = tr[tr['long_view'] == 1]
    g = lv_tr.groupby('user_id')['duration_ms'].agg(['sum', 'count'])
    gmean = lv_tr['duration_ms'].mean()
    for name, part in feats.items():
        if name == 'train':
            s, _ = _train_sc(tr, ['user_id'], 'dur_lv', past)
            c, _ = _train_sc(tr, ['user_id'], 'long_view', past)
        else:
            s = part['user_id'].map(g['sum']).fillna(0.0).to_numpy()
            c = part['user_id'].map(g['count']).fillna(0.0).to_numpy()
        m = np.where(c > 0, s / np.maximum(c, 1), gmean)
        part['u_lv_dur'] = m.astype('float32')
        part['dur_diff'] = np.abs(part['duration_ms'] - part['u_lv_dur']).astype('float32')
        part['dur_ratio'] = (part['duration_ms'] / np.maximum(part['u_lv_dur'], 1)
                             ).clip(upper=20).astype('float32')

    # user x author / user x video / user x tag1 interactions from train
    tag1 = vb.set_index('video_id')['tag'].astype(str).str.split(',').str[0]
    tag1 = pd.to_numeric(tag1, errors='coerce').fillna(-1).astype('int16')
    for name, part in feats.items():
        part['tag1'] = part['video_id'].map(tag1).fillna(-1).astype('int16')
    tr['tag1'] = tr['video_id'].map(tag1).fillna(-1).astype('int16')

    edges10 = np.quantile(tr['duration_ms'].to_numpy(), np.linspace(0, 1, 11)[1:-1])
    for name, part in feats.items():
        part['dur_bucket'] = np.searchsorted(
            edges10, part['duration_ms'].to_numpy()).astype('int16')
    tr['dur_bucket'] = np.searchsorted(
        edges10, tr['duration_ms'].to_numpy()).astype('int16')

    # user x bucket rates, shrunk toward the user's own (LOO) rate so the
    # within-user variation is pure personalization signal
    def user_cross(bucket_col, cnt_name, rate_name, prior=CROSS_PRIOR):
        g = tr.groupby(['user_id', bucket_col])['long_view'].agg(['sum', 'count'])
        for name, part in feats.items():
            pr = part['u_lv_rate'].to_numpy()
            if name == 'train':
                s, c = _train_sc(tr, ['user_id', bucket_col], 'long_view', past)
            else:
                idx = pd.MultiIndex.from_arrays([part['user_id'], part[bucket_col]])
                s = g['sum'].reindex(idx).fillna(0.0).to_numpy()
                c = g['count'].reindex(idx).fillna(0.0).to_numpy()
            part[rate_name] = ((s + prior * pr) / (c + prior)).astype('float32')
            part[cnt_name] = c.astype('float32')

    user_cross('author_id', 'ua_cnt', 'ua_lv_rate')
    user_cross('tag1', 'ut_cnt', 'ut_lv_rate')
    user_cross('dur_bucket', 'ud_cnt', 'ud_lv_rate')
    user_cross('tab', 'utab_cnt', 'utab_lv_rate')

    # exposure-only session/recency features. These read the impression stream
    # (timestamps, ids shown) of each split — never any label from the row's
    # own evaluation window — so they are legal for valid/test scoring.
    for name, part in feats.items():
        o = part[['user_id', 'video_id', 'author_id', 'tag1', 'date',
                  'time_ms']].sort_values(['user_id', 'time_ms'], kind='stable')
        u = o['user_id'].to_numpy()
        t = o['time_ms'].to_numpy(dtype='int64')
        new_u = np.concatenate([[True], u[1:] != u[:-1]])
        gap = t - np.concatenate([[0], t[:-1]])
        gap = np.where(new_u, -1, gap)
        d = o['date'].to_numpy()
        new_day = new_u | np.concatenate([[True], d[1:] != d[:-1]])
        new_sess = new_day | (gap > 30 * 60 * 1000)
        pos = np.arange(len(o))
        sess_start = np.maximum.accumulate(np.where(new_sess, pos, 0))
        part.loc[o.index, 'sess_pos'] = (pos - sess_start).astype('float32')
        part.loc[o.index, 'gap_prev_ms'] = np.log1p(
            np.where(gap < 0, np.nan, gap)).astype('float32')
        part.loc[o.index, 'impr_idx_day'] = o.groupby(
            ['user_id', 'date']).cumcount().to_numpy().astype('float32')
        for key, colname in (('video_id', 'uv_seen_today'),
                             ('author_id', 'ua_seen_today'),
                             ('tag1', 'ut_seen_today')):
            part.loc[o.index, colname] = o.groupby(
                ['user_id', 'date', key]).cumcount().to_numpy().astype('float32')

    # days since the user/video was last seen in the exposure stream
    # (train events plus the split's own strictly-earlier days)
    for key, out_col in (('user_id', 'u_days_since'), ('video_id', 'v_days_since')):
        tr_ev = tr[[key, 'dayord']].drop_duplicates()
        for name, part in feats.items():
            pool = tr_ev if name == 'train' else pd.concat(
                [tr_ev, part[[key, 'dayord']].drop_duplicates()])
            ev = pool.drop_duplicates().sort_values('dayord', kind='stable')
            left = part[[key, 'dayord']].reset_index().sort_values(
                'dayord', kind='stable')
            m = pd.merge_asof(left, ev.rename(columns={'dayord': 'ev_day'}),
                              left_on='dayord', right_on='ev_day', by=key,
                              allow_exact_matches=False)
            gap_d = (m['dayord'] - m['ev_day']).to_numpy(dtype='float64')
            part.loc[m['index'].to_numpy(), out_col] = \
                np.where(np.isnan(gap_d), 99.0, gap_d).astype('float32')

    # multi-tag user affinity: mean over the video's tags of user-tag rate
    vt = vb[['video_id', 'tag']].copy()
    vt['tag'] = vt['tag'].astype(str).str.split(',')
    vt = vt.explode('tag')
    vt['tag'] = pd.to_numeric(vt['tag'], errors='coerce').fillna(-1).astype('int16')
    tr_e = tr[['user_id', 'video_id', 'date', 'long_view']].merge(vt, on='video_id')
    gt = tr_e.groupby(['user_id', 'tag'])['long_view'].agg(['sum', 'count'])
    gpast = None
    if past:
        gpe = (tr_e.groupby(['user_id', 'tag', 'date'])['long_view']
               .agg(['sum', 'count']).reset_index()
               .sort_values(['user_id', 'tag', 'date'], kind='stable'))
        gpe['ps'] = gpe.groupby(['user_id', 'tag'])['sum'].cumsum() - gpe['sum']
        gpe['pc'] = gpe.groupby(['user_id', 'tag'])['count'].cumsum() - gpe['count']
        gpast = gpe.set_index(['user_id', 'tag', 'date'])
    for name, part in feats.items():
        e = part[['user_id', 'video_id', 'date', 'long_view']].reset_index().merge(
            vt, on='video_id')
        ridx = e['index'].to_numpy()
        pr = part['u_lv_rate'].to_numpy()[ridx]
        if name == 'train' and past:
            idx = pd.MultiIndex.from_arrays([e['user_id'], e['tag'], e['date']])
            s = gpast['ps'].reindex(idx).fillna(0.0).to_numpy()
            c = gpast['pc'].reindex(idx).fillna(0.0).to_numpy()
            rate = (s + CROSS_PRIOR * pr) / (c + CROSS_PRIOR)
            cnt = c
        elif name == 'train':
            idx = pd.MultiIndex.from_arrays([e['user_id'], e['tag']])
            s = gt['sum'].reindex(idx).fillna(0.0).to_numpy()
            c = gt['count'].reindex(idx).fillna(0.0).to_numpy()
            y = e['long_view'].to_numpy()
            rate = (s - y + CROSS_PRIOR * pr) / (c - 1 + CROSS_PRIOR)
            cnt = c - 1
        else:
            idx = pd.MultiIndex.from_arrays([e['user_id'], e['tag']])
            s = gt['sum'].reindex(idx).fillna(0.0).to_numpy()
            c = gt['count'].reindex(idx).fillna(0.0).to_numpy()
            rate = (s + CROSS_PRIOR * pr) / (c + CROSS_PRIOR)
            cnt = c
        denom = np.bincount(ridx, minlength=len(part)).astype('float64')
        rsum = np.bincount(ridx, weights=rate, minlength=len(part))
        csum = np.bincount(ridx, weights=cnt, minlength=len(part))
        part['utags_lv_rate'] = np.where(
            denom > 0, rsum / np.maximum(denom, 1),
            part['u_lv_rate']).astype('float32')
        part['utags_cnt'] = np.where(
            denom > 0, csum / np.maximum(denom, 1), 0.0).astype('float32')

    # item-item cosine CF over train long_views
    from scipy.sparse import csr_matrix

    def build_cf(pairs):
        ucodes, uuniq = pd.factorize(pairs['user_id'])
        vcodes, vuniq = pd.factorize(pairs['video_id'])
        M = csr_matrix((np.ones(len(pairs), dtype='float32'), (ucodes, vcodes)),
                       shape=(len(uuniq), len(vuniq)))
        norms = np.sqrt(np.asarray(M.sum(axis=0)).ravel())
        ninv_ = (1.0 / np.maximum(norms, 1e-9)).astype('float32')
        S_ = (M.T @ M).toarray().astype('float32')
        S_ *= ninv_[:, None]
        S_ *= ninv_[None, :]
        v2c = {v: i for i, v in enumerate(vuniq)}
        hist_ = pairs.groupby('user_id')['video_id'].agg(list)
        return S_, ninv_, v2c, hist_

    def cf_fill(part, mask, S_, ninv_, v2c, hist_, loo):
        cf = np.zeros(len(part), dtype='float32')
        hcnt = np.zeros(len(part), dtype='float32')
        pv = part['video_id'].to_numpy()
        ylv = part['long_view'].to_numpy()
        for u, rows in part.groupby('user_id').indices.items():
            if mask is not None:
                rows = rows[mask[rows]]
                if not len(rows):
                    continue
            h = hist_.get(u)
            if h is None:
                continue
            hc = np.fromiter((v2c[v] for v in h), dtype=np.int64, count=len(h))
            cc = np.fromiter((v2c.get(v, -1) for v in pv[rows]),
                             dtype=np.int64, count=len(rows))
            ok = cc >= 0
            vals = np.zeros(len(rows), dtype='float32')
            if ok.any():
                vals[ok] = S_[np.ix_(cc[ok], hc)].sum(1)
            if loo:
                # leave-user-out: a lv=1 row sits in its own history, so drop
                # the self term AND this user's +1 in every co(v, w), w in H(u)
                own = ylv[rows].astype('float32')
                sumninv = float(ninv_[hc].sum())
                ni_v = np.where(ok, ninv_[np.maximum(cc, 0)], 0.0)
                vals -= own * (1.0 + ni_v * (sumninv - ni_v))
                hcnt[rows] = len(hc) - own
            else:
                hcnt[rows] = len(hc)
            cf[rows] = vals
        return cf, hcnt

    lv_pairs = tr.loc[tr['long_view'] == 1,
                      ['user_id', 'video_id']].drop_duplicates()
    S, ninv, v2code, hist = build_cf(lv_pairs)
    if past:
        # first-half matrix scores second-half train rows; first-half rows get 0
        # midpoint of the DATE RANGE (not the row median — days are unevenly
        # sized, and a row-weighted split would shift the two-phase boundary)
        udates = np.unique(tr['date'].to_numpy())
        cut = int(udates[len(udates) // 2])
        early = tr[tr['date'] < cut]
        pairs_e = early.loc[early['long_view'] == 1,
                            ['user_id', 'video_id']].drop_duplicates()
        S_e, ninv_e, v2c_e, hist_e = build_cf(pairs_e)
    for name, part in feats.items():
        if name == 'train' and past:
            mask = part['date'].to_numpy() >= cut
            cf, hcnt = cf_fill(part, mask, S_e, ninv_e, v2c_e, hist_e, loo=False)
        elif name == 'train':
            cf, hcnt = cf_fill(part, None, S, ninv, v2code, hist, loo=True)
        else:
            cf, hcnt = cf_fill(part, None, S, ninv, v2code, hist, loo=False)
        part['cf_sum'] = cf
        part['u_hist_cnt'] = hcnt
        part['cf_mean'] = (cf / np.maximum(hcnt, 1)).astype('float32')
    del S

    # EASE item-item model (closed-form ridge on the lv co-occurrence Gram),
    # past mode only: first-half matrix scores second-half train rows,
    # full-window matrix scores valid/test. Base mode writes zeros.
    def build_ease(pairs, lam=250.0):
        ucodes, uuniq = pd.factorize(pairs['user_id'])
        vcodes, vuniq = pd.factorize(pairs['video_id'])
        M = csr_matrix((np.ones(len(pairs), dtype='float32'), (ucodes, vcodes)),
                       shape=(len(uuniq), len(vuniq)))
        G = (M.T @ M).toarray().astype('float64')
        idx = np.arange(G.shape[0])
        G[idx, idx] += lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[idx, idx] = 0.0
        v2c = {v: i for i, v in enumerate(vuniq)}
        hist_ = pairs.groupby('user_id')['video_id'].agg(list)
        # cf_fill sums S_[cand, w] over w in hist; EASE score is sum over
        # B[w, cand], so hand it B transposed
        return B.T.astype('float32'), v2c, hist_

    def ease_columns(pairs_full, pairs_early, out_sum, out_mean):
        Bt, ev2c, ehist = build_ease(pairs_full, lam=ease_lam)
        Bt_e, ev2c_e, ehist_e = build_ease(pairs_early, lam=ease_lam)
        for name, part in feats.items():
            if name == 'train':
                mask = part['date'].to_numpy() >= cut
                es, ehc = cf_fill(part, mask, Bt_e, None, ev2c_e, ehist_e,
                                  loo=False)
            else:
                es, ehc = cf_fill(part, None, Bt, None, ev2c, ehist, loo=False)
            if out_sum:
                part[out_sum] = es
            part[out_mean] = (es / np.maximum(ehc, 1)).astype('float32')

    if past:
        clk_pairs = tr.loc[tr['is_click'] == 1,
                           ['user_id', 'video_id']].drop_duplicates()
        clk_early = early.loc[early['is_click'] == 1,
                              ['user_id', 'video_id']].drop_duplicates()
        for ease_lam, cs, cm, src in (
                (250.0, 'ease_sum', 'ease_mean', (lv_pairs, pairs_e)),
                (50.0, None, 'ease50_mean', (lv_pairs, pairs_e)),
                (1000.0, None, 'ease1000_mean', (lv_pairs, pairs_e)),
                (250.0, None, 'easeclk_mean', (clk_pairs, clk_early))):
            ease_columns(src[0], src[1], cs, cm)
    else:
        for name, part in feats.items():
            for c in GROUPS['ease'] + GROUPS['ease2']:
                part[c] = np.float32(0.0)

    g = tr.groupby(['user_id', 'video_id'])['long_view'].agg(['sum', 'count'])
    for name, part in feats.items():
        if name == 'train':
            s, c = _train_sc(tr, ['user_id', 'video_id'], 'long_view', past)
        else:
            idx = pd.MultiIndex.from_frame(part[['user_id', 'video_id']])
            s = pd.Series(g['sum'].reindex(idx).to_numpy()).fillna(0.0).to_numpy()
            c = pd.Series(g['count'].reindex(idx).to_numpy()).fillna(0.0).to_numpy()
        part['uv_cnt'] = c.astype('float32')
        part['uv_lv_cnt'] = s.astype('float32')

    # --- video side (basic) ---
    vb2 = vb.set_index('video_id')
    upload = pd.to_datetime(vb2['upload_dt'], errors='coerce')
    for name, part in feats.items():
        for col in ('video_type', 'upload_type', 'visible_status', 'music_type'):
            part[col] = part['video_id'].map(vb2[col]).astype('category')
        for col in ('server_width', 'server_height', 'video_duration'):
            part[col] = pd.to_numeric(part['video_id'].map(vb2[col]),
                                      errors='coerce').astype('float32')
        part['aspect'] = (part['server_width'] / part['server_height'].clip(lower=1)
                          ).astype('float32')
        up = part['video_id'].map(upload)
        cur = pd.to_datetime(part['date'].astype(str))
        part['upload_age'] = (cur - up).dt.days.astype('float32')

    # --- video statistic side (dataset-provided aggregates) ---
    vs2 = vs.set_index('video_id')
    show = vs2['show_cnt'].clip(lower=1)
    play = vs2['play_cnt'].clip(lower=1)
    stat = pd.DataFrame(index=vs2.index)
    stat['s_show_cnt'] = np.log1p(vs2['show_cnt'])
    stat['s_play_cnt'] = np.log1p(vs2['play_cnt'])
    stat['s_play_user_num'] = np.log1p(vs2['play_user_num'])
    stat['s_play_progress'] = vs2['play_progress']
    stat['s_complete_rate'] = vs2['complete_play_cnt'] / play
    stat['s_longtime_rate'] = vs2['long_time_play_cnt'] / play
    stat['s_shorttime_rate'] = vs2['short_time_play_cnt'] / play
    stat['s_valid_rate'] = vs2['valid_play_cnt'] / play
    stat['s_like_rate'] = vs2['like_cnt'] / show
    stat['s_comment_rate'] = vs2['comment_cnt'] / show
    stat['s_share_rate'] = vs2['share_cnt'] / show
    stat['s_collect_rate'] = vs2['collect_cnt'] / show
    stat['s_download_rate'] = vs2['download_cnt'] / show
    stat['s_follow_rate'] = vs2['follow_cnt'] / show
    stat['s_reduce_rate'] = vs2['reduce_similar_cnt'] / show
    stat['s_double_click_rate'] = vs2['double_click_cnt'] / show
    stat = stat.astype('float32')
    for name, part in feats.items():
        joined = part[['video_id']].join(stat, on='video_id')
        for col in stat.columns:
            part[col] = joined[col]

    # --- user side ---
    uf2 = uf.set_index('user_id')
    for col in ('follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days'):
        uf2[col] = pd.to_numeric(uf2[col], errors='coerce').astype('float32')
    for col in ('is_lowactive_period', 'is_live_streamer', 'is_video_author'):
        uf2[col] = pd.to_numeric(uf2[col], errors='coerce').astype('float32')
    for name, part in feats.items():
        for col in GROUPS['uside']:
            v = part['user_id'].map(uf2[col])
            if col.startswith('onehot') or col == 'user_active_degree':
                part[col] = v.astype('category')
            else:
                part[col] = v.astype('float32')

    # --- raw ids as categoricals ---
    for name, part in feats.items():
        part['user_id_c'] = part['user_id'].astype('category')
        part['video_id_c'] = part['video_id'].astype('category')
        part['author_id_c'] = part['author_id'].astype('category')
        part['tab'] = part['tab'].astype('category')
        part['tag1'] = part['tag1'].astype('category')

    all_cols = ['user_id', 'video_id', 'date', 'long_view'] + \
        [c for g_, cs in GROUPS.items() if g_ not in SIDECAR for c in cs]
    for name, part in feats.items():
        part = part.sort_values('user_id', kind='stable').reset_index(drop=True)
        part[all_cols].to_parquet(os.path.join(cache_dir, f'{name}.parquet'))
        print(f"wrote {name}: {len(part)} rows, {len(all_cols)} cols")
    print(f"build done in {time.time()-t0:.1f}s (past={past})")


class OfficialFeval:
    """Vectorized mean(GAUC, nDCG@5) for early stopping (no tie correction;
    final numbers always come from the official evaluate())."""

    def __init__(self, user_ids):
        uid = pd.factorize(user_ids)[0]          # rows are user-contiguous
        self.uid = uid
        self.n = len(uid)
        self.nu = int(uid.max()) + 1
        self.counts = np.bincount(uid, minlength=self.nu).astype(np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        self.disc = (1.0 / np.log2(np.arange(5) + 2)).astype(np.float64)
        self.cumdisc = np.concatenate([[0.0], np.cumsum(self.disc)])

    def primary(self, y, s):
        uid, nu = self.uid, self.nu
        npos = np.bincount(uid, weights=y, minlength=nu)
        order = np.lexsort((-s, uid))
        u_sorted, y_sorted = uid[order], y[order]
        pos = np.arange(self.n) - self.starts[u_sorted]
        m5 = pos < 5
        dcg = np.bincount(u_sorted[m5], weights=y_sorted[m5] * self.disc[pos[m5]],
                          minlength=nu)
        idcg = self.cumdisc[np.minimum(npos, 5).astype(np.int64)]
        ndcg = np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0).mean()
        order2 = np.lexsort((s, uid))
        u2, y2 = uid[order2], y[order2]
        rank = np.arange(self.n) - self.starts[u2] + 1.0
        srank = np.bincount(u2, weights=y2 * rank, minlength=nu)
        nneg = self.counts - npos
        ok = (npos > 0) & (nneg > 0)
        auc_u = (srank - npos * (npos + 1) / 2.0) / np.maximum(npos * nneg, 1)
        gauc = float((auc_u[ok] * npos[ok]).sum() / npos[ok].sum()) if ok.any() else 0.5
        return (gauc + float(ndcg)) / 2.0

    def __call__(self, preds, ds):
        return 'primary', self.primary(ds.get_label(), preds), True


def train(a):
    global lgb
    import lightgbm as lgb
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, 'preds'), exist_ok=True)
    t0 = time.time()
    groups = a.features.split(',')
    cols = [c for g in groups for c in GROUPS[g]]
    if a.top_from:
        with open(os.path.join(OUT, f'{a.top_from}.json')) as fh:
            ref = json.load(fh)
        booster = lgb.Booster(model_str=open(
            os.path.join(OUT, f'{a.top_from}.txt')).read())
        ref_cols = [c for g in ref['features'].split(',') for c in GROUPS[g]]
        gains = dict(zip(ref_cols, booster.feature_importance('gain')))
        keep = {c for c, _ in sorted(gains.items(), key=lambda x: -x[1])
                [:a.top_n]}
        cols = [c for c in cols if c in keep]
        print(f"  feature selection: {len(cols)} of {len(ref_cols)} kept "
              f"(top {a.top_n} by gain from {a.top_from})")
    cache_dir = CACHE + ('_past' if a.past else '')
    need_date = a.mixw != 1.0
    sidecars = [SIDECAR[g] for g in groups if g in SIDECAR]
    side_cols = {c for _, c in sidecars}
    pq_cols = [c for c in cols if c not in side_cols]
    parts = {}
    for name in ('train', 'valid') + (('test',) if a.eval_test else ()):
        part = pd.read_parquet(
            os.path.join(cache_dir, f'{name}.parquet'),
            columns=['user_id', 'long_view'] + (['date'] if need_date else [])
            + pq_cols)
        for subdir, col in sidecars:
            arr = np.load(os.path.join(OUT, subdir, f'{name}.npy'))
            assert len(arr) == len(part), f"{subdir}/{name} misaligned"
            part[col] = arr.astype('float32')
        parts[name] = part
    cats = [c for c in cols if c in CATS]
    for name, part in parts.items():
        for c in cats:
            part[c] = part[c].astype('category')

    params = dict(
        objective=a.objective, learning_rate=a.lr, num_leaves=a.num_leaves,
        min_data_in_leaf=a.min_data, feature_fraction=a.ff,
        bagging_fraction=a.bagging, bagging_freq=1 if a.bagging < 1 else 0,
        lambda_l2=a.l2, num_threads=a.threads, seed=a.seed, verbosity=-1,
        max_cat_threshold=64, cat_smooth=a.cat_smooth, cat_l2=10.0,
    )
    tr, va = parts['train'], parts['valid']
    # Metric-aware weighting: GAUC skips users whose list is all-positive or
    # all-negative, and nDCG@5 is constant for them, so such rows cannot move
    # the score. Downweight the analogous degenerate (user, day) groups in
    # train. mixw = 1.0 disables this.
    w_tr = None
    if a.mixw != 1.0:
        g = tr.groupby(['user_id', 'date'])['long_view'].agg(['sum', 'count'])
        idx = pd.MultiIndex.from_frame(tr[['user_id', 'date']])
        s = g['sum'].reindex(idx).to_numpy()
        c = g['count'].reindex(idx).to_numpy()
        mixed = (s > 0) & (s < c)
        w_tr = np.where(mixed, 1.0, a.mixw).astype('float32')
        print(f"  mixed-group rows: {mixed.mean():.1%}, "
              f"degenerate weight {a.mixw}")
    if a.objective in ('lambdarank', 'rank_xendcg'):
        params.update(metric='None')
        if a.objective == 'lambdarank':
            params.update(label_gain=[0, 1],
                          lambdarank_truncation_level=a.trunc)
        grp_tr = tr.groupby('user_id', sort=False).size().to_numpy()
        grp_va = va.groupby('user_id', sort=False).size().to_numpy()
        dtr = lgb.Dataset(tr[cols], tr['long_view'], group=grp_tr,
                          weight=w_tr, categorical_feature=cats)
        dva = lgb.Dataset(va[cols], va['long_view'], group=grp_va,
                          reference=dtr, categorical_feature=cats)
    else:
        params.update(metric='None')
        dtr = lgb.Dataset(tr[cols], tr['long_view'], weight=w_tr,
                          categorical_feature=cats)
        dva = lgb.Dataset(va[cols], va['long_view'], reference=dtr,
                          categorical_feature=cats)

    feval = OfficialFeval(va['user_id'].to_numpy())
    m = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                  feval=feval,
                  callbacks=[lgb.early_stopping(a.patience, verbose=False,
                                                first_metric_only=True)])
    best_iter = m.best_iteration

    res = {'name': a.name, 'features': a.features, 'objective': a.objective,
           'past': a.past, 'mixw': a.mixw,
           'params': {k: v for k, v in params.items()
                      if k not in ('verbosity', 'num_threads')},
           'best_iter': best_iter, 'n_features': len(cols)}
    for name in parts:
        if name == 'train':
            continue
        p = m.predict(parts[name][cols], num_iteration=best_iter)
        r = evaluate(parts[name]['user_id'].tolist(),
                     parts[name]['long_view'].tolist(), p.tolist())
        res[name] = {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in r.items()}
        np.save(os.path.join(OUT, 'preds', f'{a.name}_{name}.npy'), p)
    imp = sorted(zip(cols, m.feature_importance('gain')), key=lambda x: -x[1])
    res['top_features'] = [[c, round(float(v), 1)] for c, v in imp[:25]]
    res['wall_seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, f'{a.name}.txt'), 'w') as fh:
        fh.write(m.model_to_string(num_iteration=best_iter))
    with open(os.path.join(OUT, f'{a.name}.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    v = res['valid']
    print(f"{a.name}: valid GAUC {v['GAUC']:.4f} nDCG@5 {v['nDCG@5']:.4f} "
          f"primary {v['primary']:.4f} | iter {best_iter} | {res['wall_seconds']}s")
    if a.eval_test:
        t = res['test']
        print(f"{a.name}: test  GAUC {t['GAUC']:.4f} nDCG@5 {t['nDCG@5']:.4f} "
              f"primary {t['primary']:.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--name', default='run')
    ap.add_argument('--features',
                    default='base,item,user,author,ua,uv,utag,vside,vstat,uside')
    ap.add_argument('--objective', default='binary',
                    choices=['binary', 'lambdarank', 'rank_xendcg'])
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--num_leaves', type=int, default=127)
    ap.add_argument('--min_data', type=int, default=50)
    ap.add_argument('--ff', type=float, default=0.9)
    ap.add_argument('--bagging', type=float, default=0.9)
    ap.add_argument('--l2', type=float, default=1.0)
    ap.add_argument('--cat_smooth', type=float, default=10.0)
    ap.add_argument('--trunc', type=int, default=30)
    ap.add_argument('--rounds', type=int, default=3000)
    ap.add_argument('--patience', type=int, default=100)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval_test', action='store_true')
    ap.add_argument('--top_from', default=None,
                    help='run name whose gain importances select features')
    ap.add_argument('--top_n', type=int, default=60)
    ap.add_argument('--mixw', type=float, default=1.0,
                    help='weight for degenerate (user,day) train groups; '
                         '1.0 disables metric-aware weighting')
    ap.add_argument('--past', action='store_true',
                    help='past-only stats for train rows (separate cache)')
    ap.add_argument('--stats', default=None,
                    help='LO,HI dates overriding the aggregate source pool')
    ap.add_argument('--only', default=None,
                    help='comma-separated splits to write')
    ap.add_argument('--suffix', default=None, help='cache dir suffix')
    a = ap.parse_args()
    if a.build:
        build(a.past,
              stats=tuple(int(x) for x in a.stats.split(',')) if a.stats else None,
              only=a.only.split(',') if a.only else None,
              suffix=a.suffix)
    if a.train:
        train(a)
