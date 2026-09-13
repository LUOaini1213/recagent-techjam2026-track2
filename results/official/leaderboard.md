# Official-protocol leaderboard (kuairand-starter-kit)

Label `long_view`, within-user ranking, **primary = mean(GAUC, nDCG@5)**.
Split: train 20220408–0421 / valid 0422–0428 / test 0429–0508.
All tuning on valid; the designated submission is the validation-best
checkpoint at run stop (FAQ 2.9.1c), scored once on test (2026-09-01).

| Model | valid | test |
|---|---|---|
| random (uniform noise, mean of 5 seeds — measured) | 0.4836 | primary 0.4757 |
| pop (published) | 0.5807 | 0.5715 |
| FM starter kit (published) | 0.6016 | 0.5946 |
| FM torch reimpl (`scripts/official_fm.py`) | 0.6019 | 0.5943 |
| LightGBM best single pre-EASE (`p3_ff04`) | 0.6067 | 0.5997 |
| LightGBM + EASE single (`p4_ease`) | 0.6080 | — |
| interim blend (frozen 08-31, superseded) | 0.6072 | 0.6007 |
| **FINAL: 0.80·gbdt5v2 + 0.20·rank3** | **0.6081** | **0.6015** |

Final test: GAUC 0.6697, nDCG@5 0.5333. **+0.0069 primary over the FM
baseline**. Attainable range on this split, measured by
`scripts/official_ceiling.py` into `results/official/ceiling.json`: random
0.4757 → label-perfect 0.8645, which puts the baseline at 30.6% of the range
(`position_in_range.fm_published_0.5946` = 0.3058) and this submission at 32.4%
(`final_0.6015` = 0.3236).
gbdt5v2 = 5-seed rank-average of `p4_ease` (all features incl. EASE + session
groups); rank3 = 3 lambdarank variants.

**Significance.** Paired user-level bootstrap on the test set (2,000 resamples
of 23,875 users, same draw for both systems; per-user pieces verified against
`evaluate.py` on the full sample): final − FM(reproduced 0.5943) = +0.0072
primary, 95% CI [+0.0058, +0.0086], P(Δ ≤ 0) = 0/2000. GAUC and nDCG@5
components in `results/official/bootstrap.json` (`scripts/official_bootstrap.py`).

## What mattered (in order of impact)

1. **Past-only train features** (`--past`): every aggregate feature for a train
   row uses only strictly-earlier days (prefix sums per key×date), matching the
   valid/test condition where all stats come from the past. Leave-one-out alone
   left a train/valid mismatch that made boosting peak at ~iter 30 and capped
   primary at ~0.597; past-only mode trains healthily for 300–600 iterations.
   Worth ≈ +0.005.
2. **FM score as a GBDT feature** (5-fold OOF on train, full model on
   valid/test) plus personalization cross features (user×duration-bucket,
   user×tab, user×tag multi-hot affinity, user×author, item-item cosine CF with
   full leave-user-out correction). User-constant features have zero
   within-user ranking power — personalization must vary within a user's list.
2b. **EASE item-item model as features** (+0.0013 single-model, carried to
   test): closed-form ridge on the long_view Gram matrix (λ=250, zero
   diagonal), scored with the same two-phase past-mode scheme (first-half
   matrix scores second-half train rows; full-window matrix scores
   valid/test). Deep-research also verified the label mechanism —
   `long_view = play_time ≥ min(duration, 18s)` to ~99.8% locally — though
   explicit threshold/margin features were flat on top of the full stack.
3. **Regularization**: lr 0.02, 63 leaves, feature_fraction 0.4–0.5,
   bagging 0.8, min_data 200 (flat plateau 0.6064–0.6067 across that region).
4. **Blending**: 5-seed rank-average backbone + 14% lambdarank diversity
   (+0.0007 valid, +0.0006 test). Greedy blends over many individual models
   scored higher on valid (0.6079) but were rejected as valid-overfit risk.

## What was tried and did NOT work (all logged, none in the submission)

A final exploration round tested eight further hypotheses after convergence.
Every one was negative or neutral — the plateau is measured, not assumed.

| Hypothesis | valid | vs 0.6080 |
|---|---|---|
| Tiny DIN target-attention sequence model, standalone | 0.6011 | did not enter the blend |
| …its 5-fold OOF score as a GBDT feature | 0.6076 | −0.0004 |
| Metric-aware weighting (degenerate user-day groups ×0.3) | 0.6075 | −0.0005 |
| Per-user rank transform for blending | 0.6081 | ±0 |
| Multi-λ EASE (50/1000) + click-EASE | 0.6072 | −0.0008 |
| Feature selection to top-60 by gain | 0.6058 | −0.0022 |
| CatBoost library diversity | — | abandoned at 3.1 CPU-h |
| Fresher statistics pool (rules-gray) | — | declined, see below |

**The freshness measurement.** Test rows sit 7–17 days after the train window,
so their aggregate features are stale, and extending the statistics pool into
the validation week was tempting. Rather than guess, the value of freshness was
measured using train+valid only, holding window length constant at 7 days:
stats ending immediately before valid scored **0.6066**, stats ending 7 days
earlier scored **0.6069**. Freshness is worth **−0.0003** — nothing. What
matters is window *length* (7 d 0.6066 → 14 d 0.6080). Extending into the
validation window could therefore buy at most ~+0.0005 by volume, which does
not justify the rules risk under FAQ 2.9.2. **Not done.** A useful side
finding: the train→test temporal gap is not a generalization threat.

**Rules audit.** `log_random_4_22_to_5_08_pure.csv` was measured to be exactly
co-extensive with the valid+test window with zero randomized rows inside the
training window, so any per-item quantity derived from it would be a feature
statistic over evaluation-window labels — a hard violation. It is used nowhere.
KuaiRand-Pure also has no impression-position column, so position-bias
debiasing would require a fabricated proxy; not pursued.

## Post-submission analysis (not part of the scored run)

Added after the 2026-09-01 deadline for the record. Neither changes the
submission; both are reproducible from committed artifacts.

**Significance of the delta.** See the bootstrap paragraph above:
+0.0072 primary, 95% CI [+0.0058, +0.0086], 0 of 2,000 resamples ≤ 0.

**How much the dataset's video-statistics table is worth.**
`video_features_statistic_pure.csv` ships with KuaiRand and its collection
period is not documented, so `s_longtime_rate` and the rest of the `vstat`
group cannot be date-bounded the way the logs can. It is platform-wide traffic
rather than the sampled log re-aggregated — median `show_cnt` 4,529 against a
median of 57 impressions per video across the whole 04/08–05/08 log. Retraining
the winning configuration with the whole group removed:

| Configuration | features | valid primary |
|---|---:|---:|
| `p4_ease` (as submitted) | 99 | 0.6080 |
| `p9_no_vstat` (no `vstat` group) | 83 | 0.6042 |
| FM baseline | — | 0.6016 |

The group is worth −0.0038 of validation primary; without it the model still
clears the baseline, by a smaller margin. Command:

```powershell
python scripts/official_lgbm.py --train --past --name p9_no_vstat --features base,item,user,author,ua,uv,utag,vside,uside,ud,utab,utags,itemcf,fm,sess,ease --lr 0.02 --num_leaves 63 --ff 0.4 --bagging 0.8 --min_data 200
```

## Pitfalls fixed along the way

- Item-item CF leave-user-out: subtracting self-similarity is not enough — the
  user's own +1 in every co(v,w), w∈H(u) must go too, else cf_mean dominates
  gain and valid collapses (0.5738).
- Early stopping must use the official metric (vectorized GAUC/nDCG@5 feval),
  not global AUC.
- LightGBM's C++ writer cannot save to paths with non-ASCII characters; write
  `model_to_string()` from Python instead.
- Parameterizing the two-phase EASE/CF boundary silently moved it from the
  date-range midpoint (0415) to the row-count median (0412), because days hold
  uneven row counts. Caught by re-running a known configuration: `p8_repro`
  reproduces `p4_ease` bit-for-bit (0.608008, iteration 714) only after the
  boundary was restored. Always keep one reproduction check in the sweep.

## Reproduce

```powershell
python scripts/official_lgbm.py --build --past     # past-only cache (production)
python scripts/official_fm.py                      # FM feature (OOF + full)
python scripts/official_lgbm.py --train --past --name p4_ease --features base,item,user,author,ua,uv,utag,vside,vstat,uside,ud,utab,utags,itemcf,fm,sess,ease --lr 0.02 --num_leaves 63 --ff 0.4 --bagging 0.8 --min_data 200
# seeds 1-4 with --seed N; rank members: --objective lambdarank --trunc 10 --ff 0.5 (+ seeds)
python scripts/official_predict_test.py --quiet --name <each member>
python scripts/official_aggregate.py --name gbdt5v2 --members p4_ease,p4_ease_s1,p4_ease_s2,p4_ease_s3,p4_ease_s4
python scripts/official_aggregate.py --name rank3 --members p_rank_t10,p3_rank_ff05,p3_rank_s1
python scripts/official_final.py                   # FINAL = 0.80*rank01(gbdt5v2) + 0.20*rank01(rank3) -> preds/FINAL_*.npy + final.json, scored with evaluate.py
python scripts/official_runlog.py                  # regenerate logs/official_runs.jsonl
```

Full sweep history: `results/official/*.json` (40+ runs across three
workflow-orchestrated sweeps: regularization/ablation, past-cache grid,
ff-neighborhood + seeds).
