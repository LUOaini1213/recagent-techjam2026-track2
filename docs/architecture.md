# Architecture

The agent treats KuaiRand ranking as an MLE loop, not a one-shot notebook.

```text
raw logs  ->  prepare (train-only vocab, OOV=0)
          ->  train/val parquet          test parquet (labels untouched until final)
          ->  agent catalog
                inspect  (list-length diagnosis)
                popularity (probe)
                DeepFM click (official-style baseline)
                LightGBM + item stats (explore)
                multi-task DeepFM (GPU only)
          ->  logs/runs.jsonl, results/val_best.json
          ->  scripts/final_eval_test.py  (once)
```

## Why the first cut was wrong

Val impression lists are short (median 5, 99.9% ≤ 50). Scoring `0.5 * NDCG@10 + 0.5 * Recall@50` makes almost every model look the same on half the number. Selection and early-stop now use **NDCG@10** (`primary`). Official Recall@50 is still logged for the submission table.

Other fixes:

- Categorical vocab is fit on **train only**. Unseen val/test IDs map to 0. Ranking still groups by original `eval_user_id`, otherwise all cold-start users collapse into bucket 0.
- The trainer never loads test.
- Item statistic features (`show_cnt`, `play_cnt`, `play_progress`, `like_cnt`) are joined for trees; DeepFM gets log-buckets.
- CWM is a backbone reference, not a drop-in. Original CWM drops `is_click` and trains watch time.
- CPU DeepFM uses 2 epochs and 250k subsample so the baseline exists without a GPU. Multi-task is skipped without CUDA.
- `prepare` is skipped when `data/processed/vocab.json` is already valid.

## Metric contract

| Name | Use |
|---|---|
| NDCG@10 (`primary`) | Agent selection, early stop |
| Recall@50 | Official pair, expect saturation |
| `score` | 50/50 mix for the write-up |
| NDCG@5, AUC, list-length stats | Diagnostics |
