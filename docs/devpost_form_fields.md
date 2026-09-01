# Devpost form — paste-ready content

---

## Elevator pitch

```
An agent that runs the entire ML iteration loop by itself — 23 hypotheses, zero human code, and it caught its own data leak along the way.
```

---

## About the project

*(paste everything between the rules into the Markdown editor)*

---

## Inspiration

Every ML engineer knows the loop: look at the data, build a feature, train,
evaluate, realize you were wrong, do it again. It is the biggest time sink in
recommender work and almost none of it is creative. We wanted to find out how
much of that loop a agent could actually own — not "suggest code to a human",
but propose the hypothesis, write the code, run it, read its own score, and
decide what to try next.

The honest test of that is a benchmark where you cannot fool yourself: a fixed
official metric, a strict temporal split, and a hidden test set you are only
allowed to touch once.

## What it does

RecAgent runs the full MLE loop unattended on **KuaiRand-Pure** under the
organizers' starter-kit protocol — label `long_view`, within-user ranking,
primary score = mean(GAUC, nDCG@5), date split train 04/08–04/21, validation
04/22–04/28, hidden test 04/29–05/08.

It reproduces the official baselines, then iterates: hypothesis → feature or
model change → train → score with the untouched official `evaluate.py` →
reflect → next hypothesis. Independent experiments are fanned out in parallel.
Every iteration is logged with its hypothesis, the reasoning behind it, the
change applied, and the resulting validation score.

**Result on the hidden test set: 0.6015 (GAUC 0.6697, nDCG@5 0.5333) against
the official FM baseline of 0.5946 — an improvement of +0.0069.**

That number deserves its scale. On this benchmark a perfect ranking scores only
**0.8645**, because 27% of users have no positive label at all and 9% are all
positive; random scoring sits at 0.4753. The official baseline therefore
already captures 31.0% of the attainable range, and this submission reaches
**32.4%**. The ceiling is not 1.0, and progress should be read against 0.8645.

The run finished in **23 scored iterations** of the 50 allowed and about
**3.5 hours** of active wall-clock of the 6 permitted, on **8 CPU cores with no
GPU**, with **zero manual code or hyperparameter edits** by a human.

## How we built it

Four components, driven by the agent in a loop:

1. **Feature builder** — every aggregate for a training row is computed from
   strictly earlier days via per-key×date prefix sums.
2. **Model layer** — LightGBM as the ranker, a PyTorch factorization machine
   whose out-of-fold scores become a feature, and item-item models (cosine CF
   and a closed-form EASE ridge).
3. **The official evaluator**, vendored and never modified.
4. **Reflection** — reads the scores, picks the next hypothesis, and fans out
   sweeps in parallel when experiments are independent.

The final submission is a rank-average blend: a five-seed LightGBM backbone at
0.80 and lambdarank members at 0.20, designated as the validation-best
checkpoint and scored on the hidden test exactly once.

**What actually moved the number, in order:**

- **Past-only training features (+0.005, the single biggest gain).** Originally
  the training rows used leave-one-out statistics computed over their own week,
  while validation rows saw only strictly past data. That mismatch capped
  boosting at roughly 30 rounds and pinned the score near 0.597. Recomputing
  training statistics as strictly-earlier-day prefix sums aligned the two
  conditions, and the model trained healthily for 300–700 rounds.
- **Personalization that varies inside a list.** We measured that user-constant
  features have essentially zero within-user ranking power — `u_lv_rate` alone
  scores 0.484, worse than a coin flip on this metric. Only signals that vary
  across a user's own candidates can move it: the FM out-of-fold score,
  user×duration-bucket, user×tab, user×tag affinity, and item-item scores.
- **EASE as features (+0.0013).** A closed-form ridge on the long_view Gram
  matrix with a zeroed diagonal, scored two-phase so training rows never see a
  matrix built from their own week.

## Challenges we ran into

**The agent caught its own data leak.** After adding collaborative-filtering
features the validation score collapsed from 0.5993 to 0.5738. Inspecting
feature importances showed `cf_mean` carrying 1,456,859 gain against 343,228
for the next feature — four times anything else, which is the signature of a
leak, not a discovery. The diagnosis: the user's own history was scoring the
user. Subtracting self-similarity is not enough; the user's own +1 contribution
to *every* co-occurrence `co(v, w)` for `w` in their history has to go too. The
correction was derived, applied, and the score recovered — logged as an
error→recovery pair.

**A silent confound, caught by a reproduction check.** Parameterizing the
two-phase EASE boundary accidentally moved it from the date-range midpoint
(04/15) to the row-count median (04/12), because days hold uneven row counts.
The experiment running at the time was invalidated. We caught it by re-running
a known configuration: it reproduced the reference score bit-for-bit
(0.608008, iteration 714) only after the boundary was restored. Every sweep now
carries a reproduction check.

**Knowing when to stop, and proving it.** After convergence we tested eight
more hypotheses — a target-attention sequence model both standalone and as a
feature, metric-aware sample weighting, per-user rank blending, multi-λ EASE,
feature selection, and CatBoost for library diversity. **All eight were
negative or neutral, and all eight are in the run log with their numbers.** The
plateau is measured, not assumed.

**We measured a tempting rules-gray idea instead of taking it.** Test rows sit
7–17 days after the training window, so their statistics are stale, and
extending the statistics pool into the validation week was appealing. Rather
than guess, the agent built the measurement and ran it on train+validation only,
holding window length constant at seven days: statistics ending immediately
before validation scored 0.6066, statistics ending seven days earlier scored
0.6069. **Freshness is worth −0.0003 — nothing.** What matters is window
*length* (7 days 0.6066 → 14 days 0.6080). The technique was worth at most
+0.0005, so it was declined rather than risk the rule that training data is the
train split only. The same experiment produced a reassuring side finding: the
train→test temporal gap is not a generalization threat.

We also audited `log_random_4_22_to_5_08_pure.csv` and found it exactly
co-extensive with the validation and test windows, with zero randomized rows
inside the training window. Any per-item quantity derived from it would be a
feature statistic computed over evaluation-window labels. It is used nowhere.

## Accomplishments that we're proud of

The leak detection, more than the score. A model that goes up is unremarkable;
an agent that watches its own score collapse, reads the feature importances,
names the mechanism, derives the correction and recovers — with no human in the
loop — is the part that felt like the actual goal of this track.

Second: the negative results are first-class citizens of the log. Eight failed
hypotheses with their numbers and reasoning are more useful to the next person
than a leaderboard row.

Third: the discipline held. Every decision was made on validation. The hidden
test was scored once, for the checkpoint that validation had already chosen.

## What we learned

- **Train/inference condition mismatch is worth more than model capacity.**
  The largest gain of the entire run came from fixing *when* a statistic was
  computed, not from any model change.
- **A metric's structure dictates which features can possibly help.** For a
  within-user ranking metric, a feature constant across a user's list carries
  exactly zero information, no matter how predictive it looks globally.
- **An anomalously dominant feature is a leak until proven otherwise.** Four-x
  gain over the runner-up was the tell.
- **Measure the tempting shortcut before deciding on it.** The freshness
  question could have been answered with a plausible story in either direction.
  One controlled experiment settled it in fifteen minutes and turned a
  compliance gamble into an informed pass.

## What's next

Sequence modeling is the honest gap: our target-attention model was trainable
on CPU but not competitive, and a GPU budget would let us pool last-N behavior
embeddings properly instead of feeding aggregate summaries to a GBDT. The
bonus benchmarks (KuaiRand-1k and 27k) are untouched on this hardware. And the
convergence rule was formalized mid-run — a production agent would fix it
before the first iteration.

The loop itself is not dataset-specific. The past-only discipline, the leak
detection heuristic, and the honest logging transfer to any ranking problem.

---

## Built with

*(paste as comma-separated tags)*

```
python, lightgbm, pytorch, pandas, numpy, scipy, scikit-learn, pyarrow, pytest, claude, anthropic, claude-code, kuairand, gradient-boosting, factorization-machines, lambdarank, ease, collaborative-filtering, learning-to-rank, edge-tts, ffmpeg, pillow, catboost, recommender-systems
```

---

## "Try it out" links

```
https://github.com/LUOaini1213/recagent-techjam2026-track2
```

---

## Video demo link

```
https://youtu.be/OoPYcCveBfw
```

---

## Image gallery

Upload from `docs/video/gallery/` — five 3:2 stills, all under 5 MB:

| File | What it shows |
|---|---|
| `01_results.png` | hidden-test results, with the attainable-range bar |
| `02_leak.png` | the leak: score collapse and the 4x importance anomaly |
| `03_architecture.png` | the four components and the loop |
| `04_build.png` | past-only feature build, matching the official split |
| `05_convergence.png` | the stop event and the one-shot test evaluation |

Put `01_results.png` first — it is the thumbnail on the gallery card.
