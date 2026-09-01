# Frozen LightGBM — val + one-shot hidden test

Official date split. Hidden test scored **once** after val freeze.  
Selection metric: **NDCG@10**.

## Validation (tune here)

| Model | NDCG@10 | NDCG@5 | users |
|---|---|---|---|
| popularity | 0.8239 | 0.7780 | 23354 |
| DeepFM click (bot, CPU subsample) | 0.8325 | — | 23354 |
| GBDT+CTR blend w&lt;1 | all below 0.8412 | — | 23354 |
| LightGBM (previous freeze) | 0.841210 | 0.7997 | 23354 / 19243 scored |
| gbdt_rerank (time=0, pop=−0.15) | 0.841693 | 0.7999 | 23354 / 19243 scored |
| gbdt_side (unused user/video side cols) | 0.842276 | 0.8011 | 23354 / 19243 scored |
| **gbdt_inlist_res (width/height/aspect)** | **0.842328** | **0.8013** | 23354 / 19243 scored |

## Hidden test (once)

| Model | NDCG@10 | NDCG@5 | AUC | users |
|---|---|---|---|---|
| popularity | 0.8156 | 0.7696 | — | 23199 / 18617 scored |
| **LightGBM (submission)** | **0.8341** | **0.7933** | **0.7438** | 23199 / 18617 scored |

Test Δ vs popularity: **+0.0185 NDCG@10** (val Δ was +0.0173; lift held).  
Test lists still short (median 4, 99.8% ≤50) so Recall@50 ≈ 0.9997.

Artifacts: `results/gbdt.txt`, `results/val_best.json`, `results/test_once.json`, `results/blend_val.json`.
