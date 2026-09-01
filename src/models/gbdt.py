from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fit_predict_gbdt(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    seed: int,
    out_path: Path,
) -> np.ndarray:
    import lightgbm as lgb

    feats = cat_cols + [c for c in num_cols if c in train.columns]
    x_train = train[feats]
    y_train = train["is_click"].to_numpy()
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        categorical_feature=[c for c in cat_cols if c in feats],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_path))
    return model.predict_proba(val[feats])[:, 1]


def fit_predict_gbdt_side(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    seed: int,
    out_path: Path,
    raw_dir: Path,
    vocab_path: Path,
) -> np.ndarray:
    """Same pointwise GBDT plus unused static user/video side columns."""
    import lightgbm as lgb

    from src.data.side import add_side_features

    train_f, extra_cat, extra_num = add_side_features(train, raw_dir, vocab_path)
    val_f, _, _ = add_side_features(val, raw_dir, vocab_path)
    feats = cat_cols + extra_cat + [c for c in num_cols + extra_num if c in train_f.columns]
    cats = [c for c in cat_cols + extra_cat if c in feats]
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(train_f[feats], train_f["is_click"].to_numpy(), categorical_feature=cats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_path))
    return model.predict_proba(val_f[feats])[:, 1]


def fit_predict_gbdt_inlist(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    seed: int,
    out_path: Path,
    raw_dir: Path,
    vocab_path: Path,
    mode: str = "all",
) -> np.ndarray:
    """gbdt_side plus a subset of in-list tag/resolution/time order."""
    import lightgbm as lgb

    from src.data.inlist import add_inlist_features

    train_f, extra_cat, extra_num = add_inlist_features(train, raw_dir, vocab_path, mode=mode)
    val_f, _, _ = add_inlist_features(val, raw_dir, vocab_path, mode=mode)
    feats = cat_cols + extra_cat + [c for c in num_cols + extra_num if c in train_f.columns]
    cats = [c for c in cat_cols + extra_cat if c in feats]
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(train_f[feats], train_f["is_click"].to_numpy(), categorical_feature=cats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_path))
    return model.predict_proba(val_f[feats])[:, 1]


def fit_predict_gbdt_hist(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    seed: int,
    out_path: Path,
) -> np.ndarray:
    """Pointwise LightGBM plus train-only CTR counts the frozen GBDT did not use."""
    import lightgbm as lgb

    from src.data.history import add_history_features

    train_f, val_f, hist_cols = add_history_features(train, val)
    # Keep the frozen GBDT capacity; only add signals that vary inside a user's list.
    extra = [c for c in ("item_ctr", "item_n", "time_rank", "time_z") if c in hist_cols]
    feats = cat_cols + [c for c in num_cols + extra if c in train_f.columns]
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        train_f[feats],
        train_f["is_click"].to_numpy(),
        categorical_feature=[c for c in cat_cols if c in feats],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_path))
    return model.predict_proba(val_f[feats])[:, 1]


def fit_predict_ranker(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    seed: int,
    out_path: Path,
) -> np.ndarray:
    """LambdaRank grouped by original user id. Pointwise binary GBDT does not optimize NDCG."""
    import lightgbm as lgb

    from src.data.history import add_history_features

    train_f, val_f, hist_cols = add_history_features(train, val)
    user_col = "eval_user_id" if "eval_user_id" in train_f.columns else "user_id"
    feats = cat_cols + [c for c in num_cols + hist_cols if c in train_f.columns]

    train_s = train_f.sort_values(user_col, kind="mergesort")
    group = train_s.groupby(user_col, sort=False).size().to_numpy()
    y = train_s["is_click"].to_numpy()
    x = train_s[feats]

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[5, 10],
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=96,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
        lambdarank_truncation_level=20,
    )
    model.fit(
        x,
        y,
        group=group,
        categorical_feature=[c for c in cat_cols if c in feats],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_path))
    return model.predict(val_f[feats])


def predict_gbdt(df: pd.DataFrame, model_path: Path, cat_cols: list[str], num_cols: list[str]) -> np.ndarray:
    import lightgbm as lgb

    feats = cat_cols + [c for c in num_cols if c in df.columns]
    booster = lgb.Booster(model_file=str(model_path))
    return booster.predict(df[feats])
