"""
Match-based train/test splitter for xG modeling.

Why not random split? Shots in the same match share context (same teams,
same defensive line, same weather, same referee). Splitting randomly would
leak this shared context between train and test, inflating metrics.

We use GroupShuffleSplit with match_id as the group key. We additionally
ensure the target class ratio (is_goal ~10%) is approximately preserved
in both splits via a stratified group approach.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


RANDOM_STATE = 42


def match_based_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    group_col: str = "match_id",
    target_col: str = "is_goal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by match groups, preserving approximate target stratification.

    Strategy:
    1. Compute per-match goal rate.
    2. Bin matches into quartiles of goal rate.
    3. Use GroupShuffleSplit within each bin to respect test_size.
    4. Concatenate.

    This ensures both train and test have similar overall goal rates while
    guaranteeing no match appears in both sets.

    Args:
        df: Full feature DataFrame with group_col and target_col.
        test_size: Fraction of data for test set (by number of matches).
        random_state: Reproducibility seed.
        group_col: Column identifying match groups.
        target_col: Binary target column.

    Returns:
        (train_df, test_df) tuple.
    """
    match_stats = df.groupby(group_col)[target_col].agg(["count", "mean"])
    match_stats.columns = ["n_shots", "goal_rate"]

    # Bin matches by goal rate into 4 quartiles for stratification.
    match_stats["bin"] = pd.qcut(
        match_stats["goal_rate"], q=4, labels=False, duplicates="drop"
    )

    rng = np.random.RandomState(random_state)

    test_match_ids = set()
    for _, bin_matches in match_stats.groupby("bin"):
        ids = bin_matches.index.tolist()
        rng.shuffle(ids)
        n_test = max(1, int(len(ids) * test_size))
        test_match_ids.update(ids[:n_test])

    test_mask = df[group_col].isin(test_match_ids)
    train_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    return train_df, test_df


def validate_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "match_id",
    target_col: str = "is_goal",
) -> dict:
    """Return split diagnostics."""
    train_matches = set(train_df[group_col].unique())
    test_matches = set(test_df[group_col].unique())
    overlap = train_matches & test_matches

    return {
        "train_shots": len(train_df),
        "test_shots": len(test_df),
        "train_matches": len(train_matches),
        "test_matches": len(test_matches),
        "match_overlap": len(overlap),
        "train_goal_rate": train_df[target_col].mean(),
        "test_goal_rate": test_df[target_col].mean(),
        "test_fraction": len(test_df) / (len(train_df) + len(test_df)),
    }
