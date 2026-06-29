"""
Full feature pipeline definition for xG modeling.

The stateless feature transformers (geometry, freeze frame) are applied
first to enrich the DataFrame. The OneHotEncoder for categoricals is
defined here but is intended to be fit on TRAINING data only (Phase 3),
to physically prevent data leakage.

statsbomb_xg, shot_outcome, is_goal, player, team, match_id are NEVER
included as features -- they are metadata or target.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .transformers import FreezeFrameFeatures, GeometryFeatures


CATEGORICAL_COLUMNS = [
    "shot_body_part",
    "shot_type",
    "shot_technique",
    "play_pattern",
    "kp_pass_height",
]

NUMERIC_FEATURE_COLUMNS = [
    "geom_distance",
    "geom_angle_rad",
    "geom_angle_deg",
    "geom_x_dist",
    "geom_y_dist",
    "geom_in_box",
    "ff_has_frame",
    "ff_n_opponents_in_cone",
    "ff_n_opponents_within_3m",
    "ff_n_opponents_within_5m",
    "ff_dist_nearest_opponent",
    "ff_dist_to_gk",
    "ff_gk_off_line",
    "ff_gk_y_offset",
    "ff_has_gk",
    "ff_n_teammates_in_box",
    # Key-pass (assist) numeric features
    "kp_pass_length",
    "kp_pass_angle",
    # Game state
    "score_diff",
]

BOOLEAN_FEATURE_COLUMNS = [
    "under_pressure",
    # Native StatsBomb shot flags
    "shot_open_goal",
    "shot_one_on_one",
    "shot_first_time",
    "shot_deflected",
    "shot_follows_dribble",
    "shot_aerial_won",
    "shot_redirect",
    # Key-pass boolean flags
    "kp_has_key_pass",
    "kp_is_cross",
    "kp_is_cutback",
    "kp_is_through_ball",
]

FORBIDDEN_COLUMNS = [
    "statsbomb_xg",
    "shot_outcome",
    "is_goal",
    "match_id",
    "player",
    "team",
    "competition",
    "season",
    "minute",
    "second",
    "x",
    "y",
    "shot_freeze_frame",
]


def build_stateless_features_pipeline() -> Pipeline:
    """Pipeline that only computes the stateless features.

    Used in scripts/build_features.py to materialize a feature parquet
    before the train/test split.
    """
    return Pipeline([
        ("geometry", GeometryFeatures()),
        ("freeze_frame", FreezeFrameFeatures()),
    ])


def build_categorical_encoder() -> ColumnTransformer:
    """ColumnTransformer for one-hot encoding categorical columns.

    Will be fit on TRAINING data in Phase 3.
    """
    return ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_full_feature_pipeline() -> Pipeline:
    """End-to-end feature pipeline: stateless + categorical encoding.

    NOTE: this pipeline MUST be fit only on training data. The OneHotEncoder
    is the only stateful component; geometry/freeze_frame are stateless.
    """
    return Pipeline([
        ("stateless", build_stateless_features_pipeline()),
        ("categorical", build_categorical_encoder()),
    ])
