"""
Baseline xG model: Logistic Regression with full feature pipeline.

This is the interpretable reference model. Gradient boosting (Phase 4)
will be compared against this baseline.

Design decisions:
- class_weight=None: xG is a probability problem, not a ranking problem.
  Adjusting class weights distorts calibration.
- C=1.0: default L2 regularization. Tuning is not the point here.
- Pipeline ensures no leakage: OHE and imputer fit only on train.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.pipeline import (
    BOOLEAN_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
)


def get_feature_columns() -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, boolean, categorical) column lists for model input."""
    return NUMERIC_FEATURE_COLUMNS, BOOLEAN_FEATURE_COLUMNS, CATEGORICAL_COLUMNS


def build_baseline_pipeline() -> Pipeline:
    """Build the full Logistic Regression pipeline.

    Structure:
        ColumnTransformer:
            - numeric: impute median -> standard scale
            - boolean: passthrough (already 0/1)
            - categorical: OHE with handle_unknown='ignore'
        -> LogisticRegression (L2, C=1.0)
    """
    numeric_cols, boolean_cols, categorical_cols = get_feature_columns()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "boolean",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ]),
                boolean_cols,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight=None,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])

    return pipeline


def get_lr_coefficients(pipeline: Pipeline, feature_names: Optional[list] = None) -> pd.DataFrame:
    """Extract LR coefficients for interpretability.

    Args:
        pipeline: Fitted pipeline with 'preprocessor' and 'classifier' steps.
        feature_names: Optional explicit feature names. If None, tries to infer.

    Returns:
        DataFrame with feature names and coefficients, sorted by absolute value.
    """
    lr = pipeline.named_steps["classifier"]
    coefs = lr.coef_[0]

    if feature_names is None:
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(coefs))]

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    return df
