"""
Player-level xG aggregation and finishing skill analysis.

Finishing skill = actual_goals - xG_sum (goals above expected).
Positive: overperformer (clinical finisher).
Negative: underperformer (wasteful).

Confidence band: binomial standard error for goals_above_xg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_player_table(
    df: pd.DataFrame,
    model_xg_col: str = "model_xg",
    target_col: str = "is_goal",
    player_col: str = "player",
    min_shots: int = 20,
) -> pd.DataFrame:
    """Aggregate per-player shooting stats and finishing signal.

    Args:
        df: Shot-level DataFrame with model_xg, is_goal, player columns.
        model_xg_col: Column containing model xG predictions.
        target_col: Binary goal column.
        player_col: Player name column.
        min_shots: Minimum shots to include player.

    Returns:
        DataFrame sorted by goals_above_xg descending, columns:
        player, shots, actual_goals, xg_sum, xg_per_shot,
        actual_conversion, goals_above_xg, goals_above_xg_se, sb_xg_sum.
    """
    agg = df.groupby(player_col).agg(
        shots=(target_col, "count"),
        actual_goals=(target_col, "sum"),
        xg_sum=(model_xg_col, "sum"),
        sb_xg_sum=("statsbomb_xg", "sum"),
    ).reset_index()

    agg = agg[agg["shots"] >= min_shots].copy()

    agg["xg_per_shot"] = agg["xg_sum"] / agg["shots"]
    agg["actual_conversion"] = agg["actual_goals"] / agg["shots"]
    agg["goals_above_xg"] = agg["actual_goals"] - agg["xg_sum"]

    # Standard error: sqrt(n * p * (1-p)) where p = xg_per_shot
    # This is the std dev of total goals under expected distribution
    agg["goals_above_xg_se"] = np.sqrt(
        agg["shots"] * agg["xg_per_shot"] * (1.0 - agg["xg_per_shot"])
    )

    agg = agg.sort_values("goals_above_xg", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1

    return agg[
        [
            "rank", "player", "shots", "actual_goals", "xg_sum", "sb_xg_sum",
            "xg_per_shot", "actual_conversion", "goals_above_xg", "goals_above_xg_se",
        ]
    ]
