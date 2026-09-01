# RecAgent — Autonomous ML Research Agent for KuaiRand-Pure (Track 2)

## What it does

RecAgent is an autonomous ML research agent that runs the full MLE loop on the
official KuaiRand-Pure benchmark (label `long_view`, primary = mean(GAUC,
nDCG@5), starter-kit protocol): it reproduces the official baselines, then
iterates hypothesis → feature/code change → train → official-metric eval →
reflect, entirely agent-driven, logging every iteration.

**Result: hidden-test primary 0.6015 (GAUC 0.6697, nDCG@5 0.5333) vs the
official FM baseline 0.5946 — +0.0069**, with the run staying inside the
track's hard caps (23 scored iterations of the 50 allowed, ~3.5 h active
wall-clock of the 6 h budget, CPU only) and strict test-label hygiene
(validation-only tuning; the designated submission is the validation-best
checkpoint per FAQ 2.9.1c, scored once on test).

For scale: on this benchmark a perfect ranking scores only 0.8645 (27% of
users have no positive label) and random scores 0.4753, so the FM baseline
already captures 31% of the attainable range and this submission reaches
32.4%.

## How it addresses the problem statement

- **Reproduce the baseline**: torch FM reimplementation matches the published
  numbers (valid 0.6015 vs 0.6016; test within the published seed std).
- **Iterate autonomously**: the agent proposed and tested ~45 hypotheses in
  three workflow-orchestrated sweeps plus targeted iterations. Highlights it
  found on its own:
  - *Past-only train features*: aligning train rows to the valid/test
    condition (stats from strictly earlier days via per-key×date prefix sums)
    fixed a train/valid mismatch that capped boosting at iteration ~30
    (+0.005 primary, the single biggest step).
  - *Within-user signal analysis*: measured that user-constant features have
    zero within-user ranking power, and pivoted to personalization features
    (FM OOF score, user×duration/tab/tag crosses, item-item CF, EASE).
  - *Leak-hunting*: detected a leave-user-out leak in the CF feature from a
    4× feature-gain anomaly, derived the exact correction, and verified
    recovery — recorded as an error→recovery pair in the run log.
  - *EASE item-item model* (closed-form ridge) as features: +0.0013 single
    model, found and integrated in one iteration on CPU.
- **Converged result**: final submission is a rank-average blend of a 5-seed
  LightGBM backbone with a lambdarank diversity member, designated as the
  validation-best checkpoint per FAQ 2.9.1(c).

## The demo video is also built by code

`docs/video/recagent_demo.mp4` (3:00, 1080p, narrated, subtitled) is produced
by three scripts, not by a screen recorder: narration is synthesized with a
neural TTS voice and its measured durations drive every scene length;
subtitles are cut from the engine's own word-boundary events; the picture is
rendered from real numbers in `results/official/` and `logs/`; and assembly
ends in a hard assertion that deletes the file if it misses 3:00. Editing
`docs/video/narration.json` and rerunning three commands regenerates it.

## Tech stack

- **Development tools**: Claude Code (agent harness), VS Code, PowerShell/
  Windows 11, 8-core CPU / 16 GB RAM (no GPU used).
- **APIs**: Anthropic Claude (Fable 5) as the reasoning engine driving the
  loop; no other external APIs.
- **Libraries**: LightGBM 4.7, PyTorch (CPU) for the FM feature model,
  pandas / NumPy / SciPy / PyArrow, scikit-learn, pytest.
- **Datasets**: KuaiRand-Pure only (official splits from the starter kit;
  `log_random` used for nothing; KuaiRand-1k/27k not used).

## Run logs & reproduction

- Per-iteration log (hypothesis, why, change, valid GAUC/nDCG@5, errors &
  recovery, interventions, wall-clock): `logs/official_runs.jsonl`.
- Reproduction: `results/official/leaderboard.md` (commands for every stage);
  official metric computed only by the untouched starter-kit `evaluate.py`.
- Human interventions: 3 goal-level directives, 0 manual code/parameter edits.

## What we tried that failed (and how we know)

After convergence the agent tested eight more hypotheses — a target-attention
sequence model (standalone and as a feature), metric-aware sample weighting,
per-user rank blending, multi-λ EASE, feature selection, CatBoost library
diversity, and a fresher statistics pool. All eight were negative or neutral,
and all are in the run log with their numbers. Two are worth calling out:

- **We measured a rules-gray-area idea instead of taking it.** Extending the
  feature-statistics pool into the validation week would have given test rows
  fresher statistics. Instead of guessing, the agent measured what freshness is
  worth using train+valid only, at constant window length: **−0.0003**. Volume,
  not recency, is what matters. The technique was worth at most +0.0005, so it
  was declined rather than risk FAQ 2.9.2. The same experiment showed the
  train→test temporal gap is not a generalization threat.
- **A silent confound, caught by a reproduction check.** Parameterizing the
  EASE/CF two-phase boundary moved it from the date midpoint to the row-count
  median. Re-running a known configuration reproduced the reference score
  bit-for-bit only after the fix, invalidating one experiment that had to be
  rerun.

## Limitations & next steps

- Sequence modeling is shallow: exposure-only session features were flat, and
  a CPU budget ruled out trained sequence models (DIN/SASRec-class); with a
  GPU we would pool last-N behavior embeddings under target attention.
- The convergence rule was formalized mid-run (declared in the log header);
  a production agent would fix it before the first iteration.
- Bonus benchmarks (KuaiRand-1k/27k) not attempted on this hardware.
