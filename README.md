# RecAgent — Autonomous ML Research Agent

> **TikTok TechJam 2026 result: this entry did not place.** The competition has
> concluded and this submission was not among those selected. Everything below is
> what was built and measured — each number traces to a committed artifact, and
> none of it was ever a placement claim.

**TikTok TechJam 2026 · Track 2 · KuaiRand-Pure (`long_view` ranking, official starter-kit protocol)**

[![ci](https://github.com/LUOaini1213/recagent-techjam2026-track2/actions/workflows/ci.yml/badge.svg)](https://github.com/LUOaini1213/recagent-techjam2026-track2/actions/workflows/ci.yml)

[▶ 3-minute demo video](https://youtu.be/OoPYcCveBfw) ·
[Per-iteration run log](logs/RUN_LOG.md) (table; [the JSONL it is rendered from](logs/official_runs.jsonl)) ·
[Leaderboard and reproduction](results/official/leaderboard.md) ·
[Devpost description](docs/devpost_description.md)

An agent that runs the MLE loop itself — inspect → feature → train → evaluate →
reflect — on the official KuaiRand-Pure benchmark, and beats the organizers' FM
baseline on the hidden test set without a human writing or editing a line of
model code.

## Result

Scoring is the untouched starter kit: `kuairand-starter-kit/evaluate.py`, label
`long_view`, within-user ranking, **primary = mean(GAUC, nDCG@5)**. Validation
was used for every decision; the hidden test split was scored **once**, on the
validation-best checkpoint at run stop (FAQ 2.9.1c).

| Model | valid | test |
|---|---:|---:|
| random (uniform noise, mean of 5 seeds — measured here) | 0.4836 | 0.4757 |
| pop (published) | 0.5807 | 0.5715 |
| FM starter kit (published) | 0.6016 | 0.5946 |
| **RecAgent final blend** | **0.6081** | **0.6015** |

**+0.0069 test primary over the FM baseline** (test GAUC 0.6697, nDCG@5 0.5333).
For scale: a label-perfect ranking scores 0.8645 on this benchmark (27% of users
have no positive label), so on the 0.4757 → 0.8645 attainable range the baseline
sits at **30.6%** and this submission at **32.4%**. Every figure in that sentence
is a field of [`results/official/ceiling.json`](results/official/ceiling.json),
which `scripts/official_ceiling.py` writes from the labels. Final recipe:
`results/official/final.json`.

Is the delta more than noise? A paired user-level bootstrap (2,000 resamples of
the 23,875 test users, both systems scored on the same draw) puts
final − FM at **+0.0072** primary against our FM reproduction (0.5943), 95% CI
**[+0.0058, +0.0086]**; 0 of 2,000 resamples were ≤ 0. Per-user pieces are
checked against `evaluate.py` on the full sample before resampling —
`scripts/official_bootstrap.py`, artifact
[`results/official/bootstrap.json`](results/official/bootstrap.json).

Budget actually used, against the track's hard caps:

| | used | cap |
|---|---:|---:|
| scored iterations | **23** | 50 |
| active wall-clock | **~3.5 h** | 6 h |
| GPU | none (8-core CPU, 16 GB RAM) | — |
| manual code / feature / hyperparameter edits | **0** | — |

## Verify the headline without retraining

No dataset download, no training — everything the checks need is committed:

```bash
pip install -r requirements.txt
python -m pytest tests                       # 33 tests on synthetic data (2 need the raw dataset and skip)
python scripts/official_final.py --check     # re-derive FINAL_*.npy from the group predictions, then rescore it
python scripts/official_runlog.py --check    # recompile the run log from results/official/*.json
python scripts/official_runlog_md.py --check # re-render logs/RUN_LOG.md from that log
```

All four run in CI on every push, and the second one is the headline itself:
it rebuilds `FINAL_*.npy` as `0.80·rank01(gbdt5v2) + 0.20·rank01(rank3)` and
checks it bit for bit, then scores that blend with the untouched
`kuairand-starter-kit/evaluate.py` and compares GAUC / nDCG@5 / primary with
`final.json` to 5e-7. Change the `0.601486` in `final.json` by one digit and
CI goes red.

Scoring needs labels, and they are committed too:
[`results/official/labels_valid_test.npz`](results/official/labels_valid_test.npz)
— 140 KB, the `user_id` and `long_view` columns of the 124,909 valid and
170,588 test rows, in the row order of the prediction files. It is derived, not
authoritative, so `scripts/official_labels.py --check` rebuilds it from the
public KuaiRand-Pure release (`scripts/download_kuairand.py`, md5-pinned) and
fails on a single differing byte; that runs in the `labels` workflow. Editing
the npz to manufacture a score fails first against the SHA-256 digests recorded
in `scripts/official_labels.py`.

The other committed artifacts are the two group predictions and the final blend
per split (`results/official/preds/`, 7.1 MB) and `logs/official_runs.jsonl`.
The trained boosters are not (20–25 MB each; the pipeline rebuilds them).

## What "autonomous" means here

The loop was driven by an LLM agent (Claude, run through Claude Code). It
proposed each hypothesis, wrote the feature and training code, ran the official
evaluator, read the numbers and decided what to try next. The training code it
produced is deterministic Python (LightGBM, a PyTorch-CPU FM); the agent is the
thing that changed it between iterations.

Humans gave goal-level directives only — five in total, all quoted in the log
footer ("status?", "optimize to the best", "check the official standard online",
"do it", "write the demo script"). Zero manual edits to code, features or
parameters. Every scored iteration is one line in `logs/official_runs.jsonl`
with the hypothesis, why, the change applied, valid GAUC / nDCG@5 / primary,
errors and recovery, and wall-clock; seed replicates of one hypothesis are
grouped into that iteration (40+ individual training runs sit in
`results/official/*.json`).

Three iterations worth reading in the log:

1. **Past-only train features** (the single biggest step, ≈ +0.005). Every
   aggregate feature for a train row uses only strictly earlier days, so train
   rows see the same conditions valid/test rows do. Leave-one-out alone left a
   train/valid mismatch that made boosting peak at iteration ~30.
2. **A leak the agent caught and fixed itself** (iteration 2). Adding item-item
   CF and cross features *dropped* validation to 0.5738. The agent read the
   feature gains, saw `cf_mean` at 4× the next feature, derived the exact
   leave-user-out correction and recovered — logged as an error → recovery pair.
3. **A rules-gray-area idea it measured instead of taking.** Extending the
   statistics pool into the validation week would give test rows fresher
   statistics. Measured at constant window length using train+valid only,
   freshness was worth **−0.0003**; window *length* is what matters. Worth at
   most +0.0005, not worth the FAQ 2.9.2 risk. Declined.

After convergence the agent tested eight more hypotheses (sequence model,
metric-aware weighting, per-user rank blending, multi-λ EASE, feature
selection, CatBoost diversity, statistics freshness). All eight were negative
or neutral and all are in the log — the plateau is measured, not assumed.

## Method (the final recipe)

5-seed LightGBM backbone on past-only features — FM out-of-fold score,
item-item cosine CF with a full leave-user-out correction, EASE item-item
scores, user × duration-bucket / tab / tag crosses, session features — blended
by rank average with three lambdarank diversity members
(`0.80 · gbdt5v2 + 0.20 · rank3`). What each part was worth, in order of
impact, is in [`results/official/leaderboard.md`](results/official/leaderboard.md).

Rules audit: `log_random_4_22_to_5_08_pure.csv` is co-extensive with the
valid+test window and is used nowhere; KuaiRand-1k / 27k are not used; test
labels never touched model selection, early stopping or feature statistics.

## Reproduce

Python 3.11, CPU is enough. The official evaluator is vendored, unmodified, in
`kuairand-starter-kit/`.

```bash
git clone https://github.com/LUOaini1213/recagent-techjam2026-track2
cd recagent-techjam2026-track2
pip install -r requirements.txt
python scripts/download_kuairand.py          # KuaiRand-Pure from Zenodo -> data/raw/KuaiRand-Pure/data
python -m pytest tests                        # 33 tests, synthetic data only
```

Published baselines (from the starter-kit folder):

```bash
cd kuairand-starter-kit
python baseline.py --data_dir ../data/raw/KuaiRand-Pure/data --model fm
cd ..
```

The scored pipeline, stage by stage (every command is also in the leaderboard
file, next to the number it produced):

```bash
python scripts/official_lgbm.py --build --past     # past-only feature cache
python scripts/official_fm.py                      # FM feature (5-fold OOF on train, full model on valid/test)
python scripts/official_lgbm.py --train --past --name p4_ease \
  --features base,item,user,author,ua,uv,utag,vside,vstat,uside,ud,utab,utags,itemcf,fm,sess,ease \
  --lr 0.02 --num_leaves 63 --ff 0.4 --bagging 0.8 --min_data 200
# seeds 1-4: add --seed N.  rank members: --objective lambdarank --trunc 10 --ff 0.5 (+ seeds)
python scripts/official_predict_test.py --quiet --name <each member>
python scripts/official_aggregate.py --name gbdt5v2 --members p4_ease,p4_ease_s1,p4_ease_s2,p4_ease_s3,p4_ease_s4
python scripts/official_aggregate.py --name rank3  --members p_rank_t10,p3_rank_ff05,p3_rank_s1
python scripts/official_final.py                   # 0.80·gbdt5v2 + 0.20·rank3 -> preds/FINAL_*.npy + final.json, scored with evaluate.py
python scripts/official_bootstrap.py               # paired user bootstrap of (final - FM) on test -> bootstrap.json
python scripts/official_ceiling.py                 # label-perfect and random reference points -> ceiling.json
python scripts/official_labels.py                  # (user_id, long_view) of valid/test -> labels_valid_test.npz
python scripts/official_runlog.py                  # regenerate logs/official_runs.jsonl from results/official/*.json
python scripts/official_runlog_md.py               # render it as logs/RUN_LOG.md
```

Trained boosters (20–25 MB each) are not committed; the scripts rebuild them.
`official_runlog.py` compiles the run log from the result artifacts, and
`official_runlog.py --check` recompiles it in memory and diffs, so the log
cannot drift from the numbers it cites without CI saying so. Same for
`official_runlog_md.py --check` and the rendered table.

## Repository map

| Path | Role |
|---|---|
| `kuairand-starter-kit/` | official baseline and evaluator, **unmodified** |
| `scripts/official_*.py` | the scored pipeline: feature cache, FM feature, LightGBM / lambdarank members, test prediction, aggregation, blend, reference points, labels, run-log compiler. Four of them take `--check` and run in CI |
| `results/official/` | every sweep run (`*.json`), `leaderboard.md`, `final.json`, `ceiling.json`, the six prediction `.npy` and the labels `.npz` the checks score against |
| `logs/official_runs.jsonl` | the required per-iteration log (header: protocol, convergence rule, caps; footer: resource use); `logs/RUN_LOG.md` is the same log as a table |
| `docs/devpost_description.md`, `docs/video/` | submission text; the demo video and the three scripts that generate it from real repository numbers |
| `agent/`, `src/`, `configs/`, `docs/history/BOT.md` | the v0 harness from before the starter kit was published — see below |
| `tests/` | 33 tests on synthetic data: the official evaluator vs the vectorized early-stopping metric, past-only train statistics, the blend recipe, the committed labels' digests, plus the v0 leak guards and split hygiene |

## History: the pre-starter-kit protocol (superseded, kept for the record)

The first days of the hackathon predate the official starter kit. The
`agent/loop.py` catalog runner (`python scripts/run_bot.py`) targeted a sketch
of the task — `is_click` label, NDCG@10 / Recall@50, train-only vocab — and
produced a LightGBM that scored hidden-test **NDCG@10 0.8341** vs popularity
0.8156 (`results/real_val_leaderboard.md`). Those numbers are on a different
label and metric and are **not comparable** to the official results above; they
are not the submission. Once the organizers published `kuairand-starter-kit/`,
everything above was rebuilt on its protocol (see the note at the top of
`PROBLEM.md`).

## Errata

Discrepancies found in the submitted material after the deadline are listed in
[`docs/ERRATA.md`](docs/ERRATA.md) rather than quietly corrected — the
convergence rule's minimum-iteration floor contradicting the stop event, the
rule having been formalized mid-run, the hidden test having been scored twice
in the project's history (both events are in the log), the stale sections in
the submitted README, and the attainable-range floor that the submitted
material quoted as 0.4753 while the only measurement in the repository says
0.4757. No metric in `final.json` or the run log changed; the one field of the
log that did change after the deadline is named in ERRATA item 5.

## Limitations

- Sequence modeling is shallow: exposure-only session features were flat, and
  the CPU budget ruled out trained sequence models (a small DIN-style model
  scored 0.6011 standalone and did not enter the blend).
- The convergence rule (ε = 0.0005, N = 4, ≥ 30 iterations) was formalized
  mid-run and is declared as such in the log header; a production agent would
  fix it before the first iteration.
- Bonus benchmarks (KuaiRand-1k / 27k) were not attempted on this hardware.

## Demo video

`docs/video/recagent_demo.mp4` (3:00, narrated, subtitled) is generated by
`scripts/video_tts.py` → `video_scenes.py` → `video_assemble.py` from
`docs/video/narration.json` and the real numbers in `results/official/` and the
run log; the assembler deletes any cut that misses 3:00. Details in
[`docs/video/README.md`](docs/video/README.md).
