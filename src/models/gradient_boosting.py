"""
Gradient Boosting xG model using LightGBM.

LightGBM chosen over XGBoost for:
- Native categorical feature support (no OHE needed for trees)
- Faster training (histogram-based)
- Similar or better accuracy on tabular data

Calibration is checked post-training; if needed, isotonic regression is applied.
"""

from __future__ import annotations

from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.features.pipeline import (
    BOOLEAN_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)


def get_lgbm_feature_columns() -> list[str]:
    """All feature columns for LightGBM (numeric + boolean + categorical)."""
    return NUMERIC_FEATURE_COLUMNS + BOOLEAN_FEATURE_COLUMNS + CATEGORICAL_COLUMNS


def prepare_lgbm_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare DataFrame for LightGBM: select columns, set dtypes.

    LightGBM handles categoricals natively via 'category' dtype.
    NaN in numeric is handled natively by LightGBM (no imputation needed).
    """
    feature_cols = get_lgbm_feature_columns()
    X = df[feature_cols].copy()

    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype("category")

    for col in BOOLEAN_FEATURE_COLUMNS:
        X[col] = X[col].astype(int)

    return X, feature_cols


def get_default_params() -> dict:
    """Default LightGBM params tuned for xG (calibration-friendly)."""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Optional[dict] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM model.

    If validation set provided, uses early stopping.
    """
    if params is None:
        params = get_default_params()

    model = lgb.LGBMClassifier(**params)

    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["callbacks"] = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ]

    model.fit(X_train, y_train, **fit_params)
    return model


def cv_tune_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    param_grid: Optional[list[dict]] = None,
) -> tuple[dict, float]:
    """Simple grid search with GroupKFold CV.

    Returns best params and best mean log-loss.
    """
    if param_grid is None:
        param_grid = [
            {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31, "min_child_samples": 50},
            {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "min_child_samples": 30},
            {"n_estimators": 1000, "learning_rate": 0.02, "max_depth": 7, "num_leaves": 63, "min_child_samples": 30},
            {"n_estimators": 1500, "learning_rate": 0.01, "max_depth": 8, "num_leaves": 127, "min_child_samples": 20},
            {"n_estimators": 1000, "learning_rate": 0.02, "max_depth": 5, "num_leaves": 31, "min_child_samples": 50, "subsample": 0.7, "colsample_bytree": 0.7},
        ]

    from sklearn.metrics import log_loss as sk_log_loss

    gkf = GroupKFold(n_splits=n_splits)
    best_score = float("inf")
    best_params = param_grid[0]

    for i, params_candidate in enumerate(param_grid):
        full_params = get_default_params()
        full_params.update(params_candidate)

        fold_scores = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**full_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            y_pred = model.predict_proba(X_vl)[:, 1]
            fold_scores.append(sk_log_loss(y_vl, y_pred))

        mean_score = np.mean(fold_scores)
        print(f"  Config {i+1}/{len(param_grid)}: logloss={mean_score:.6f} "
              f"(params: {params_candidate})", flush=True)

        if mean_score < best_score:
            best_score = mean_score
            best_params = full_params

    return best_params, best_score


def calibrate_model(
    model: lgb.LGBMClassifier,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Post-hoc calibration using isotonic regression or Platt scaling.

    Uses cv=None with a prefitted estimator (sklearn >= 1.9 API).
    """
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=None,
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated
