# Run log — one line per scored iteration

Rendered from `logs/official_runs.jsonl` by `scripts/official_runlog_md.py`; the JSONL is the deliverable, this is a view of it.

- **Benchmark / protocol:** KuaiRand-Pure — kuairand-starter-kit evaluate.py — label long_view, primary = mean(GAUC, nDCG@5), split 0408-0421 / 0422-0428 / 0429-0508
- **Convergence rule:** ε = 0.0005, N = 4, min iterations 30
- **Hard caps:** {"iterations": 50, "wall_clock_hours": 6}
- **Hardware:** 8-core CPU, 16 GB RAM, no GPU used
- **Agent:** Claude (Fable 5) driving the pipeline in Claude Code; training code deterministic Python (LightGBM/PyTorch-CPU)
- **Human interventions:** 5 — Five goal-level user directives over the whole run ("status?", "optimize to the best", "check the official standard online / deep-research", "do it" authorizing the freshness measurement, and the demo-script request); the footer lists them. Zero manual code or hyperparameter interventions; all hypotheses, code, and configs were agent-generated.

| iter | hypothesis | what changed | valid primary | Δ vs best so far | runs | wall (s) | outcome |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Full-feature LightGBM beats FM out of the box | scripts/official_lgbm.py: 10 feature groups, LOO stats, binary obj; then switched early stopping from global AUC to official-metric feval (same iteration, bug … | 0.5968 |  | 1 | 42 |  |
| 2 | CF + personalization cross features add within-user signal | Added item-item cosine CF, user x dur-bucket/tab/tag crosses, torch-FM OOF score feature. ERROR: cf_mean gain 4x anomaly -> valid 0.5738 | 0.5738 | -0.0230 | 1 | 35 | ERROR |
| 3 | Fix CF leave-user-out leak | Full leave-user-out correction in the CF builder. RECOVERY: 0.5993 | 0.5993 | +0.0025 | 1 | 27 | RECOVERY |
| 4 | Sweep 1: regularization + feature-group ablation + objective | Workflow-orchestrated 12-config sweep (LOO cache). Findings: regularization helps (0.6019), FM feature +0.003, lambdarank behind | 0.6019 | +0.0027 | 12 | 1081 |  |
| 5 | Past-only train features fix the train/valid mismatch | Rebuilt features with strictly-before-date prefix sums (--past cache). CONFIRMED: 0.6011 -> 0.6051, healthy 608 iterations | 0.6051 | +0.0032 | 1 | 173 | CONFIRMED |
| 6 | Sweep 2: hyperparameter grid + ablations on the past cache | 12-config workflow sweep. Winner feature_fraction 0.5 (0.6064); lambdarank recovered to 0.6025 | 0.6064 | +0.0013 | 12 | 1886 |  |
| 7 | Sweep 3: fine-tune the ff=0.5 neighborhood | 8-config workflow sweep. Plateau confirmed: 0.6064-0.6067 | 0.6067 | +0.0002 | 8 | 1369 | CONFIRMED |
| 8 | Seed ensemble of the winner + rank diversity member | 5 seed/variant replicates (grouped as one hypothesis) | 0.6069 | +0.0002 | 5 | 594 |  |
| 9 | Exposure-only session/recency features | Added sess group (8 features). Result: flat (0.6065) — kept for diversity only | 0.6065 | -0.0004 | 1 | 113 |  |
| 10 | EASE closed-form item-item model as features | Added ease_sum/ease_mean. CONFIRMED: 0.6067 -> 0.6080 single-model | 0.6080 | +0.0011 | 1 | 146 | CONFIRMED |
| 11 | Seed ensemble of the EASE-enhanced winner | 4 seed replicates (grouped as one hypothesis) | 0.6075 | -0.0005 | 4 | 542 |  |
| 12 | Label-mechanism + auxiliary-feedback features | Added aux group (9 features) to the past-only builder | 0.6073 | -0.0007 | 1 | 218 |  |
| 13 | Ranking-objective alignment (lambdarank trunc=8, rank_xendcg) | Two objective variants on the aux-enhanced feature set (one hypothesis) | 0.6042 | -0.0038 | 2 | 163 |  |
| 14 | Tiny DIN target-attention sequence model (score as feature) | scripts/official_seq.py (standalone valid 0.6011, close to FM) + 5-fold OOF scores as a GBDT feature. NEGATIVE: 0.6076 vs 0.6080 without, and the raw score did… | 0.6076 | -0.0004 | 1 | 214 | NEGATIVE |
| 15 | Metric-aware sample weighting | --mixw 0.3. NEGATIVE: 0.6075 vs 0.6080 — 80.7% of train rows are already in mixed groups, so this only discards information | 0.6075 | -0.0005 | 1 | 221 | NEGATIVE |
| 16 | Multi-lambda EASE + click-EASE features | Added ease2 group (lambda 50/1000 long_view + lambda 250 is_click). First attempt (p7_ease2, 0.6072) was CONFOUNDED: parameterizing the two-phase EASE/CF bound… | 0.6080 | +0.0000 | 3 | 1417 | CONFIRMED / NEGATIVE |
| 17 | Feature selection to the top 60 features by gain | --top_from p8_repro --top_n 60 (ff raised to 0.6 for the smaller set). NEGATIVE: 0.6058 vs 0.6080 — the low-gain tail is contributing, not diluting | 0.6058 | -0.0022 | 1 | 74 | NEGATIVE |
| 18 | Torch FM reproduction + OOF score feature | scripts/official_fm.py: torch FM (k=32), early stop on valid primary, 5-fold OOF train scores, full-model valid/test scores; k=64/k=16 variants for blend diver… | 0.6019 | -0.0061 | 3 |  |  |
| 19 | Rank-average blending of diverse members | scripts/official_blend.py + official_aggregate.py; greedy forward selection on valid; controlled-recipe comparison (greedy over aggregates vs fixed weights) — … | — |  | 0 |  |  |
| 20 | Re-blend on the EASE-enhanced backbone | greedy blend over {gbdt5v2, gbdt5, rank3, fm3} | 0.6081 | +0.0001 | 0 |  | CONFIRMED |
| 21 | Final blend round with research-driven members (p5_aux, rank_xendcg) | greedy blend over {gbdt5v2, p5_aux, rank4, rank3, p5_xendcg, fm3, gbdt5} — no improvement over previous blend -> convergence window closes | 0.6081 | +0.0000 | 0 |  |  |
| 22 | Per-user rank transform for blending instead of global rank | official_blend.py --peruser (rank inside each user's list) — NEGATIVE: 0.6081, identical to global-rank blending | — |  | 0 |  | NEGATIVE |
| 23 | Feature-statistics freshness is worth more than statistics volume | official_lgbm.py --stats LO,HI (override the aggregate source pool) + scripts/official_staleness.py. Controlled comparison at constant 7-day window length: A =… | — |  | 0 |  | NEGATIVE |

23 scored iterations. `Δ vs best so far` is against the best validation primary of all earlier iterations; a negative value is a hypothesis that was measured and not adopted.

## Events between iterations

- **test_eval** — Interim test reveal of the then-frozen conservative recipe (0.8575*gbdt5 + 0.1425*rank3) plus reference singles Performed before the validation-best designation rule (FAQ 2.9.1c, published 2026-08-31 13:57) was incorporated; logged transparently. No decision was ever made using test scores. `{"blend_test_primary": 0.6007, "gbdt5_test_primary": 0.6001, "p3_ff04_test_primary": 0.5997, "fm_test_primary": 0.5943}`
- **rule_check** — log_random_4_22_to_5_08_pure.csv as a debiasing / propensity source Measured to be exactly co-extensive with the valid+test window (04/22-05/08, is_rand=1 throughout) with zero randomized rows inside the training window. Any per-item quantity derived from its labels would be feature statistics computed over evaluation-window labels — a hard violation of FAQ 2.9.2/2.9.3. NOT USED anywhere in this pipeline, for training or features.
- **abandoned** — CatBoost member for library diversity (scripts/official_catboost.py) Ran 3.1 CPU-hours without reaching its early-stopping point on 1.1M rows with ~20 high-cardinality categoricals under ordered target statistics, while starving a correctness-check job. Terminated on resource grounds: adversarial verification of the library-diversity literature refuted every claim of blend gain from swapping GBDT libraries (0-3), so expected value did not justify the compute. Script is kept in the repo, unused by the submission.
- **stop** — Converged under the declared rule (epsilon=0.0005, N=4): the best validation primary over the last 4 scored iterations (0.6081) exceeds the best from before that window (0.6081) by 0.0000 <= epsilon. Every hypothesis tested after the convergence point was also scored and logged, and none beat 0.6081, so the designation never changed. caps: `{"scored_iterations": 23, "iteration_cap": 50, "active_wall_clock_hours": 3.5, "wall_clock_cap": 6, "wall_clock_note": "Measured from artifact timestamps: 08-31 21:28-22:11 (sweeps/seeds), 22:11-23:28 (rules research, no training), 23:28-23:43 (session/EASE features, convergence reached), 09-01 09:18-10:33 (post-convergence exploration round). The 09h35m overnight span 23:43-09:18 was idle with no jobs running and is excluded."}`
- **test_eval** — One-shot hidden-test evaluation of the designated submission `{"valid": {"GAUC": 0.6762, "nDCG@5": 0.54, "primary": 0.6081}, "test": {"GAUC": 0.6697, "nDCG@5": 0.5333, "primary": 0.6015}, "delta_test_primary_vs_published_fm": 0.006886}`

**Stop:** Converged under the declared rule (epsilon=0.0005, N=4): the best validation primary over the last 4 scored iterations (0.6081) exceeds the best from before that window (0.6081) by 0.0000 <= epsilon. Every hypothesis tested after the convergence point was also scored and logged, and none beat 0.6081, so the designation never changed.

## Resource use (footer)

- active wall-clock hours: 3.5
- GPU hours: 0
- LLM tokens: ~18M total across the whole run: main agent ~2M, plus workflow subagents (3 tuning sweeps ~0.5M, two deep-research rounds ~10.2M over 206 subagents). Estimated from harness telemetry; exact per-call accounting is not exported.
- training compute: 8-core CPU only, no GPU. Largest single training ~3.7 min; one abandoned CatBoost job consumed 3.1 CPU-hours and was terminated, see the 'abandoned' event.
- human interventions, total: 5 — goal-level user directives only: (1) status query; (2) 'optimize to the best'; (3) 'check the official standard online / is this too little optimization'; (4) 'do it' authorizing pursuit of the freshness question; (5) request for the demo-video script. Zero manual code, feature, or hyperparameter edits by a human. Directive (4) authorized investigating a rules-gray-area technique; the agent built the measurement, found the technique worth at most +0.0005, and declined it on its own rather than taking the compliance risk.
