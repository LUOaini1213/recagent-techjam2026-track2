# Errata

Discrepancies found in the submitted material after the 2026-09-01 deadline,
listed here rather than quietly corrected. The submission itself — the frozen
`main` branch, the Devpost entry and the published video — is left as it was
submitted. Nothing below changes a score: `results/official/final.json` and
`logs/official_runs.jsonl` carry the same numbers they carried at submission,
and `scripts/official_final.py --check` re-derives them from the committed
predictions.

## 1. The convergence rule's minimum-iteration floor contradicts the stop event

`logs/official_runs.jsonl` declares `{"epsilon": 0.0005, "N": 4,
"min_iterations": 30}` and its stop event records `"scored_iterations": 23`.
The run therefore did not satisfy the floor it declared.

The ε/N part of the rule was satisfied: the best validation primary over the
last four scored iterations (0.6081) exceeded the best before that window by
0.0000. The `30` was written while thinking of individual training runs — there
are 56 of them in `results/official/*.json` — whereas "scored iteration" is
defined in the same log as *one hypothesis*, with seed replicates and sweep
members grouped. Nobody reconciled the two units before the log was compiled.
The floor should have read either "30 training runs" or "20 scored iterations".

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

## 5. Human-intervention count: 3 in the log header, 5 in the footer

The header was written when the log was first compiled and counted three
directives; two more followed (the "do it" that authorized the
feature-freshness measurement, and the request for the demo-video script). The
footer's `human_interventions_detail` quotes all five and is the correct count.
`docs/devpost_description.md` on `main` carries the stale 3. Both numbers refer
to goal-level directives; manual edits to code, features or parameters were
zero in every version.

## 6. Stale numbers in the shooting script

`docs/demo_video_script.md` was written before the final round and said
"seventeen iterations" and "4.5 h". Corrected to 23 and ~3.5 h active. The
rendered video and `docs/video/narration.json` always said 23.

## 7. The public README describes two protocols at once

`main`'s README states the official `long_view` / mean(GAUC, nDCG@5) protocol
and the 0.6015 result at the top, then continues with sections written before
the organizers published the starter kit: `is_click`, NDCG@10 / Recall@50
called the "Official pair", the 0.8239 → 0.8423 progression, and an entry point
(`scripts/run_bot.py` → `agent/loop.py`) that runs that superseded protocol and
does not reproduce the submitted result. Those sections are historical.

The rewritten, coherent README exists on branch `v2/readme-coherence` and was
deliberately not pushed during judging, so that the repository judges examine
stays byte-identical to what was submitted.

## What is *not* an erratum

- `agent/loop.py` and `scripts/run_bot.py` are genuinely part of the project's
  history (the pre-starter-kit harness), not stray files. They simply are not
  the submission, and `main` fails to say so.
- The 0.6015 / 0.6081 numbers, the 23 iterations, the 0 manual code edits and
  the resource figures are unchanged from submission.
