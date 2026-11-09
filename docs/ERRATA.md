# Errata

Discrepancies found in the submitted material after the 2026-09-01 deadline,
listed here rather than quietly corrected.

Two of the three artefacts are dated, external and left exactly as submitted:
the Devpost entry, and the published video (youtu.be/OoPYcCveBfw, rendered
2026-09-01). The third is this repository, and it is not frozen — but the
submitted tree is still in it and can be checked out. `main` runs back past the
deadline: its root commit is `78873a3` (2026-09-01 03:14:57 UTC, 161 files),
followed the same morning by `32f752d` (03:20:02) and `f0ed066` (03:22:17).
`f0ed066` is the tree as it stood at submission; everything after it is
post-deadline work, beginning with `45b3105` and `42fc934` on 2026-09-08, so
`git diff f0ed066 HEAD` is the whole of what has changed since. Each item below
that has since been corrected on `main` says so, and says what the submitted
version said.

No score moved. `results/official/final.json` and `logs/official_runs.jsonl`
carry the metrics they carried at submission, and
`scripts/official_final.py --check` re-derives the submission from the
committed group predictions and rescores it against `final.json` with the
official evaluator. What did change in the log after the deadline, all of it in
`45b3105`: the header's `human_interventions` (3 → 5) and the wording of its
`human_interventions_note` — item 5 — plus a `wall_seconds_total` field on 17
of the 23 iteration lines, which the compiler sums from the per-run
`wall_seconds` that the `results/official/*.json` already carried at
submission. `final.json` likewise only gained fields — a `reproduce` command
and the `artifacts_sha256` block — with no value edited. Nothing else in either
file differs from `f0ed066`.

## 1. The convergence rule's minimum-iteration floor contradicts the stop event

`logs/official_runs.jsonl` declares `{"epsilon": 0.0005, "N": 4,
"min_iterations": 30}` and its stop event records `"scored_iterations": 23`.
The run therefore did not satisfy the floor it declared.

The ε/N part of the rule was satisfied: the best validation primary over the
last four scored iterations (0.6081) exceeded the best before that window by
0.0000. The `30` was written while thinking of individual training runs — 56 of
them from the run itself, one per file carrying a `best_iter` in
`results/official/` (57 such files today: the extra `p9_no_vstat` is a
post-deadline diagnostic, described in the leaderboard) — whereas "scored
iteration" is defined in the same log as *one hypothesis*, with seed replicates
and sweep members grouped. Nobody reconciled the two units before the log was
compiled. The floor should have read either "30 training runs" or "20 scored
iterations".

This is a drafting error in the deliverable, not a change of stopping point:
the run stopped because ε/N were met and no later hypothesis beat 0.6081.

## 2. The convergence rule was formalized mid-run

Stated in the log header itself and in the README's limitations, but worth
repeating here: FAQ 2.9.1(a) asks for the rule to be fixed *before* the run.
Ours was written during it and applied to the tail. The header also records the
counterfactual — under the organizers' default (ε = 0.002, N = 3) the run would
have stopped earlier, at the 0.6072 validation blend.

## 3. The hidden test set was scored twice in the project's history

Both events are in the log. An interim reveal on 08-31 scored the then-frozen
recipe `0.8575·gbdt5 + 0.1425·rank3` at test primary 0.6007, together with
three reference singles; it happened before the organizers published the
validation-best designation rule (FAQ 2.9.1c, 08-31 13:57). The final one-shot
evaluation of the designated checkpoint gave 0.6015.

Everything that changed between the two — session features, EASE, five new
seeds, the re-blend to 0.80/0.20 — was selected on validation, and each of
those runs has a validation-only record in `results/official/`. The claim that
no test number influenced a decision rests on that record and on the log's
disclosure; we cannot prove a negative beyond it.

## 4. The published video misstates when the rule was fixed

At about 2:12 the narration says convergence was declared "under a rule it
fixed in advance". That contradicts item 2 above. `docs/video/narration.json`
is corrected in the repository; the published cut (youtu.be/OoPYcCveBfw) is
left as submitted.

## 5. Human-intervention count: the submitted log said 3 in the header, 5 in the footer

The header was written when the log was first compiled and counted three
directives; two more followed (the "do it" that authorized the
feature-freshness measurement, and the request for the demo-video script). The
footer's `human_interventions_detail` quotes all five and is the correct count.
Both numbers refer to goal-level directives; manual edits to code, features or
parameters were zero in every version.

Corrected since: the header of `logs/official_runs.jsonl` (and therefore the
`HEADER` dict in `scripts/official_runlog.py` that it is compiled from), its
footer, `logs/RUN_LOG.md` line 10 and `docs/devpost_description.md` all read 5.
The per-iteration `"human_interventions": 0` on every iteration line is a
different field and is unchanged.

## 6. Stale numbers in the shooting script

`docs/demo_video_script.md` was written before the final round and said
"seventeen iterations" and "4.5 h". Corrected to 23 and ~3.5 h active. The
rendered video and `docs/video/narration.json` always said 23.

## 7. The submitted README described two protocols at once

The README as submitted stated the official `long_view` / mean(GAUC, nDCG@5)
protocol and the 0.6015 result at the top, then continued with sections written
before the organizers published the starter kit: `is_click`, NDCG@10 /
Recall@50 called the "Official pair", the 0.8239 → 0.8423 progression, and an
entry point (`scripts/run_bot.py` → `agent/loop.py`) that runs that superseded
protocol and does not reproduce the submitted result. A reader had no way to
tell which of the two sets of numbers was the submission.

Corrected since: the README on `main` is the rewritten one. It states a single
protocol, and the pre-starter-kit harness now lives in one "History" section
that labels its numbers superseded and not comparable.

## 8. The attainable-range floor: 0.4753 quoted, 0.4757 measured

Every piece of submitted material put the random floor of the attainable range
at **0.4753** and derived "the baseline captures 30.7% of the range". Nothing in
this repository produces 0.4753. The starter kit publishes only the popularity
and FM baselines (`kuairand-starter-kit/baseline_scores.json`), and
`PROBLEM.md` lists the same two; the only random score any artifact here
contains is the one `scripts/official_ceiling.py` measures — uniform noise
scored by the official evaluator, mean of 5 seeds — which is **0.4757** on test
and 0.4836 on valid.

On the measured floor the two percentages are 30.6% and 32.4%, and those are
the numbers `results/official/ceiling.json` carries
(`position_in_range`: 0.3058 and 0.3236). The README, the leaderboard, the
Devpost text in this repository and the video scripts now quote 0.4757 / 30.6%
/ 32.4%, so each of them can be checked against that file. The conclusion is
unaffected — the difference between the two floors is 4e-4, a tenth of a
percentage point of range position.

Separately, the published video renders **"31.0% of the attainable range ->
32.4%"** on the results card (`docs/video/gallery/01_results.png` is a still of
it). 31.0% is wrong on either floor: 0.4753 gives 30.7% and 0.4757 gives 30.6%.
`scripts/video_scenes.py` is corrected; the published cut and the gallery still
are left as submitted, as with item 4.

## 9. "Scored once", "~45 hypotheses" and one lambdarank member

The submitted README, leaderboard and Devpost text said in bold or in the
opening paragraph that the hidden test split was "scored once", the video
narration that the agent "scores the hidden test set exactly once", and the
"scored once" wording outlived item 3 above on `main` until the November 2026
review fixes. What was scored once is the designated checkpoint; the split was revealed twice
(item 3). The same review found two smaller drifts in the Devpost text: "~45
hypotheses" (no artifact counts hypotheses; 56 in-run training runs
carry a `best_iter`, 57 with the post-deadline `p9_no_vstat`; see item 1) and "a lambdarank diversity member" (the final blend
has three, `rank3`, at weight 0.20). The leaderboard's blending line quoted the
superseded interim recipe's gain (14%, +0.0007 valid / +0.0006 test).

Corrected since: `README.md`, `results/official/leaderboard.md`,
`docs/devpost_description.md`, `docs/devpost_form_fields.md`,
`docs/demo_video_script.md`, `docs/video/narration.json` and
`scripts/video_scenes.py`. The Devpost entry and the published video are left
as submitted, as with items 4 and 8, and so are the committed
`docs/video/voiceover.mp3`, `voiceover.srt` and `timing.json`, which still
carry the old sentence because they were not regenerated from the corrected
`narration.json`. `tests/test_readme_numbers.py` now fails
if "scored once" returns without the two-reveal qualifier.

## What is *not* an erratum

- `agent/loop.py` and `scripts/run_bot.py` are genuinely part of the project's
  history (the pre-starter-kit harness), not stray files. They simply are not
  the submission — which the README's History section now states and the
  submitted README did not.
- The 0.6015 / 0.6081 numbers, the 23 iterations, the 0 manual code edits and
  the resource figures are unchanged from submission.
