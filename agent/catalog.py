"""Ordered hypotheses.

role=probe  — sanity, not the official baseline
role=baseline — official-style DeepFM click ranker; deltas are vs this
role=explore — attempts to beat the baseline

requires_gpu: skipped on CPU to keep the 72h loop tractable on a 1650/CPU box.
DeepFM still runs on CPU with subsampled rows (see configs/default.yaml).
"""

CATALOG = [
    {
        "id": "inspect",
        "model": None,
        "role": "probe",
        "requires_gpu": False,
        "hypothesis": "Read the frozen split and impression-list lengths before training.",
        "why": "Val lists are short; Recall@50 will saturate and must not drive model selection.",
    },
    {
        "id": "popularity",
        "model": "popularity",
        "role": "probe",
        "requires_gpu": False,
        "hypothesis": "Bayesian-smoothed item CTR is a cheap ranking prior.",
        "why": "Confirms the metric pipeline. Not the official baseline.",
    },
    {
        "id": "deepfm_click",
        "model": "deepfm",
        "role": "baseline",
        "requires_gpu": False,
        "hypothesis": "CWM-style DeepFM + click BCE is the official-style baseline.",
        "why": "Track 2 points at CWM backbones; the scored label is click, not watch time.",
    },
    {
        "id": "gbdt",
        "model": "gbdt",
        "role": "explore",
        "requires_gpu": False,
        "hypothesis": "LightGBM on the same cats plus item statistics beats a capacity-limited DeepFM.",
        "why": "Val ranking is a small tabular problem per user; trees fit the 4GB/CPU budget.",
    },
    {
        "id": "multitask",
        "model": "multitask",
        "role": "explore",
        "requires_gpu": True,
        "hypothesis": "Shared-bottom multi-task (click + like + long_view) transfers sparse engagement.",
        "why": "Track 2 appendix A.3. Skipped on CPU because it does not change the data/eval contract.",
    },
]
