# Grok bot brief — TikTok TechJam 2026 Track 2

You are taking over a **working** RecAgent repo. Do not rewrite the pipeline. **Run the remaining catalog on the real KuaiRand-Pure val set, pick val-best by NDCG@10, write logs.**

## Do this (in order)

```powershell
cd <this-repo>
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
python -m pytest tests -q
python scripts/run_bot.py
```

`scripts/run_bot.py` already: checks real (non-synthetic) processed data, `--fresh` wipes stale synthetic agent logs, then runs `python -m agent.loop --fresh`.

Linux/mac:

```bash
export PYTHONPATH=.
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_bot.py
```

## Hard rules

- Positive label: `is_click`. Reported metrics: NDCG@10 and Recall@50.
- **Select / early-stop by `primary` = NDCG@10.** Recall@50 saturates (val list median length 5).
- Train + val only. **Do not** run `scripts/final_eval_test.py` until the user asks. Never tune on test.
- Vocab is train-only; ranking groups by `eval_user_id`.
- CUDA torch is **optional**. CPU is enough for popularity + LightGBM. DeepFM on CPU is 2 epochs / 250k rows (may take 15–60 min). Multitask skips without CUDA.

## Known full-val number (real data, 23354 users)

Popularity item CTR: **NDCG@10 ≈ 0.8239**, Recall@50 ≈ 0.9998, AUC ≈ 0.6845.

Stale files (`logs/runs.jsonl`, old `val_best.json`) were from **40-user synthetic** data. Ignore them; `run_bot.py --fresh` replaces them.

## What is not done yet

1. LightGBM on full real val (highest ROI).
2. Fresh DeepFM click baseline on real val (CPU subsample is OK).
3. Optional: blend item CTR with GBDT/DeepFM scores if a single model does not beat 0.8239 NDCG@10.
4. Multitask DeepFM only if `torch.cuda.is_available()`.

## Done means

- `logs/runs.jsonl` has inspect + popularity + deepfm + gbdt on **n_users ≈ 23000**, not 40.
- `results/val_best.json` `metrics.primary` is NDCG@10.
- `results/gbdt_val.json` exists.
- README / results table updated with the real-val numbers.
- Test still **not** touched.

## Repo map

| Path | Role |
|---|---|
| `src/eval/metrics.py` | official metrics; `primary` = NDCG@10 |
| `src/data/prepare.py` | date split + train-only vocab |
| `src/train.py` | `--model popularity\|deepfm\|gbdt\|multitask` |
| `agent/loop.py` | catalog runner |
| `configs/default.yaml` | split, CPU DeepFM caps |
| `PROBLEM.md` | agent-readable constraints |
