# 3-Minute Demo Video — Script, Voiceover & Shot List

**Track 2 — Autonomous ML Research Agent for Recommender Systems**
Total runtime 3:00. Narration ~430 words (≈150 wpm, natural pace).
Subtitles: `docs/demo_video_subs.srt` (burn in or load as a track).

Recording tips: record narration first, then screen capture to fit. For the
demo segment, pre-run the pipeline and replay the terminal/log so nothing
stalls on camera. Keep terminal font ≥16pt — judges may watch on a laptop.

---

## 0:00–0:15 — Problem

**Voiceover**
> Every machine learning engineer knows this loop. Look at the data. Build a
> feature. Train. Evaluate. Realize you were wrong. Do it again. It is the
> single biggest time sink in recommender system work — and almost none of it
> is creative. It is iteration.

**Visuals**
- Open on the MLE loop diagram from the problem statement (inspect → feature →
  train → evaluate → reflect), animated as a spinning cycle.
- Overlay a clock or iteration counter ticking up to suggest days of work.
- Lower third: *"KuaiRand-Pure · long_view ranking · GAUC + nDCG@5"*.

---

## 0:15–0:35 — Our Solution

**Voiceover**
> So we built an agent that runs that loop by itself. It reads the problem,
> reproduces the official baseline, then proposes its own hypotheses, writes
> the code, trains, scores itself on the official metric, and decides what to
> try next. Seventeen iterations. Zero lines of code written by a human.

**Visuals**
- Cut to the agent architecture title card.
- Type-on text of three claims as they are spoken:
  *"Proposes its own hypotheses" / "Writes its own code" / "Zero human code edits"*.
- Hold on the run-log file `logs/official_runs.jsonl` scrolling past.

---

## 0:35–0:55 — Architecture

**Voiceover**
> The agent drives four components. A feature builder that computes every
> statistic strictly from the past. A model layer — gradient boosting, a
> factorization machine, and item-item models. The official evaluator, which we
> never modify. And a reflection step that reads the scores and picks the next
> hypothesis. When experiments are independent, it fans them out in parallel.

**Visuals**
- Architecture diagram, boxes lighting up in narration order:
  `Feature Builder (past-only)` → `Models: LightGBM · FM · CF/EASE` →
  `Official evaluate.py (untouched)` → `Reflect & Select`.
- Draw the loop-back arrow from Reflect to Feature Builder.
- Show the parallel sweep fan-out: one box splitting into four workers.

---

## 0:55–2:20 — Live Demo ⭐

**Voiceover**
> Here is a full run. The agent starts from raw KuaiRand-Pure logs and builds
> its feature cache. Every aggregate for a training row uses only strictly
> earlier days — that discipline alone was our single biggest gain.
>
> Now it trains and scores itself with the organizers' evaluator. Watch the log:
> each iteration records a hypothesis, the reason, the code change, and the
> validation score.
>
> This is the moment that matters. The agent adds collaborative-filtering
> features and the score collapses. It inspects feature importance, sees one
> feature with four times the gain of anything else, and diagnoses a leak — the
> user's own history was scoring the user. It derives the correction, reruns,
> and recovers. No human touched that.
>
> It keeps going. It fans out parallel sweeps. It tests a sequence model, a
> metric-aware weighting scheme, and a freshness experiment — and it records
> all three as failures, because they were.
>
> Finally it declares convergence under the rule it fixed in advance, names its
> best validation checkpoint, and scores the hidden test set exactly once.

**Visuals** (85 seconds — keep it moving, 6 beats)
1. `0:55` Terminal: `python scripts/official_lgbm.py --build --past` — rows
   counting up, "build done".
2. `1:08` Split into three panes showing train / valid / test row counts
   matching the official kit exactly (1,141,112 / 124,909 / 170,588).
3. `1:18` Training scroll, then the eval line printing
   `GAUC … nDCG@5 … primary …`.
4. `1:32` **Leak beat.** Split screen: score dropping to 0.5738 on the left,
   feature-importance bar chart on the right with `cf_mean` towering over the
   rest. Circle it. Then the corrected code diff, then the recovered score.
   *Give this beat a full 20 seconds — it is the strongest thing in the video.*
5. `1:55` Parallel sweep view: four workers running configs simultaneously,
   leaderboard rows appearing.
6. `2:08` The stop event in the log, then the single hidden-test evaluation
   printing its result.

---

## 2:20–2:45 — Results

**Voiceover**
> The official factorization-machine baseline scores 0.5946 on the hidden test.
> We reach 0.6015. That is plus 0.0069 — and on this benchmark the ceiling is
> not one point zero. A perfect ranking scores 0.8645, because a quarter of
> users have no positive label at all. Against that real range, we move the
> baseline's share from thirty-one percent to thirty-two point four. All of it
> on eight CPU cores, in four and a half hours.

**Visuals**
- Results table, rows appearing one at a time:
  `pop 0.5715` / `FM 0.5946` / `Ours 0.6015`.
- Then a horizontal range bar: `random 0.4753 ——— FM ——— Ours ——— ceiling 0.8645`
  with our marker sliding into place. **This visual is what makes the delta read
  as meaningful rather than small — do not skip it.**
- Bottom strip: `17 iterations · 0 manual code edits · CPU only · 4.5 h`.

---

## 2:45–3:00 — Impact

**Voiceover**
> Nothing here is specific to one dataset. The loop, the leak detection, the
> past-only discipline, and the honest logging of what failed — that is a
> template for any ranking problem. This is what it looks like when the
> iteration is automated and the engineer gets to think instead. Thank you.

**Visuals**
- Zoom out from the KuaiRand result to a row of dataset icons (other
  benchmarks) to suggest portability.
- End card: project name, GitHub URL, `results/official/leaderboard.md`.
- Hold the end card for the final 3 seconds in silence.

---

## Numbers to keep on screen (all verifiable in the repo)

| Claim | Where |
|---|---|
| test primary 0.6015 (GAUC 0.6697, nDCG@5 0.5333) | `results/official/final.json` |
| FM baseline 0.5946 test | `results/official/official_kit_baselines.json` |
| valid 0.6081 | `results/official/final.json` |
| 17 scored iterations, caps 50 / 6 h | `logs/official_runs.jsonl` |
| leak: 0.5738 → 0.5993 recovery | `logs/official_runs.jsonl` iterations 2–3 |
| row counts 1,141,112 / 124,909 / 170,588 | build output, matches starter kit |
