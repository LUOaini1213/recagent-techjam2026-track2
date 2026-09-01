# Track 2 constraints (agent-readable)

Autonomous ML research agent for TikTok TechJam 2026 Track 2
("Autonomous Machine Learning Research Agent for Recommender Systems").

> Historical note: an earlier draft of this file described an is_click /
> NDCG@10 protocol sketched before the official starter kit was released.
> The official protocol below (from the organizers' kuairand-starter-kit and
> the TechJam 2026 Information Document, FAQs updated 2026-08-31) supersedes it.

## Task

Beat the official FM baseline on KuaiRand-Pure under the starter-kit protocol.

- Relevance label: `long_view` (native 0/1 column)
- Metrics: GAUC and nDCG@5; **primary = mean of the two** (`kuairand-starter-kit/evaluate.py`, do not modify)
- Within-user ranking over logged impressions only
- Official baselines: pop valid 0.5807 / test 0.5715; FM valid 0.6016 / test 0.5946
- Final ranking: absolute per-metric improvement over the baseline on the
  hidden test set, equal-weighted, continuous scoring

## Split (date-based)

- Train: `log_standard_4_08_to_4_21_pure.csv` (20220408–20220421)
- Validation: 20220422–20220428 of `log_standard_4_22_to_5_08_pure.csv`
- Hidden test: 20220429–20220508 of the same file. Test labels may never be
  used — not for training, model selection, early stopping, thresholds, or
  feature statistics. Enforced by code/log review. Evaluate test once, at the end.

## Allowed

- Any open-source library, papers, public methods, pretrained weights
  (not trained on these benchmarks' test labels)
- Changes to any pipeline stage — features, model, loss, training, evaluation loop

## Forbidden

- `log_random_4_22_to_5_08_pure.csv` as training data (covers the valid/test
  window; EDA-only)
- KuaiRand-1k / 27k as auxiliary training data for Pure (temporal leak one
  level removed); they are separate bonus benchmarks only
- Tuning anything on the hidden test split

## Run rules (FAQ 2.9.1)

- Convergence rule (ε, N, optional min-iteration floor) must be fixed before
  the run and recorded in the run log. Organizers' default: ε = 0.002, N = 3,
  cumulative window over scored iterations.
- Hard caps per benchmark: **50 iterations, 6 h wall-clock**. Crashed
  iterations count toward the caps but do not advance the convergence window.
- The scored submission is the **validation-best checkpoint** at the point the
  run stops, evaluated once on the hidden test set.

## Logging (required deliverable)

Each iteration must record: hypothesis and why, code/config change applied,
resulting valid GAUC / nDCG@5 / primary, errors and recovery, human
interventions, token use, wall-clock (GPU-hours if any; wall-clock is scored).

## Hardware

Local machine, 8-core CPU, 16 GB RAM (GTX 1650 4GB unused — CPU LightGBM /
small torch models). Reference pipeline needs no GPU.
