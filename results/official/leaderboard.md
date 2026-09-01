# Official-protocol leaderboard (kuairand-starter-kit)

Label `long_view`, within-user ranking, **primary = mean(GAUC, nDCG@5)**.
Split: train 20220408–0421 / valid 0422–0428 / test 0429–0508.
All tuning on valid; the designated submission is the validation-best
checkpoint at run stop (FAQ 2.9.1c), scored once on test (2026-09-01).

| Model | valid | test |
|---|---|---|
| random (published) | — | primary 0.4753 |
| pop (published) | 0.5807 | 0.5715 |
| FM starter kit (published) | 0.6016 | 0.5946 |
| FM torch reimpl (`scripts/official_fm.py`) | 0.6019 | 0.5943 |
| LightGBM best single pre-EASE (`p3_ff04`) | 0.6067 | 0.5997 |
| LightGBM + EASE single (`p4_ease`) | 0.6080 | — |
| interim blend (frozen 08-31, superseded) | 0.6072 | 0.6007 |
| **FINAL: 0.80·gbdt5v2 + 0.20·rank3** | **0.6081** | **0.6015** |

Final test: GAUC 0.6697, nDCG@5 0.5333. **+0.0069 primary over the FM
baseline** (attainable range per organizers: random 0.4753 → label-perfect
0.8645; the baseline sits at 30.7% of that range, this submission at 32.4%).
gbdt5v2 = 5-seed rank-average of `p4_ease` (all features incl. EASE + session
groups); rank3 = 3 lambdarank variants.

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
# FINAL = 0.80*rank01(gbdt5v2) + 0.20*rank01(rank3) — results/official/final.json
python scripts/official_runlog.py                  # regenerate logs/official_runs.jsonl
```

Full sweep history: `results/official/*.json` (40+ runs across three
workflow-orchestrated sweeps: regularization/ablation, past-cache grid,
ff-neighborhood + seeds).
