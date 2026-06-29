"""
Scikit-learn compatible Transformer classes for xG feature engineering.

These transformers are stateless (their fit is a no-op) because all
operations are pure functions of the input row. They are wrapped as
sklearn Transformers so they can plug into Pipeline / ColumnTransformer
and benefit from get_feature_names_out, cross-validation, etc.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .geometry import (
    angle_to_goal,
    angle_to_goal_degrees,
    distance_to_goal,
    is_in_box,
    x_distance_to_goal_line,
    y_distance_to_goal_center,
)
from .freeze_frame import compute_all_freeze_frame_features


class GeometryFeatures(BaseEstimator, TransformerMixin):
    """Add geometric features derived from shot (x, y) coordinates.

    Adds:
        geom_distance
        geom_angle_rad
        geom_angle_deg
        geom_x_dist
        geom_y_dist
        geom_in_box

    Requires input DataFrame columns: x, y
    """

    OUTPUT_COLS = [
        "geom_distance",
        "geom_angle_rad",
        "geom_angle_deg",
        "geom_x_dist",
        "geom_y_dist",
        "geom_in_box",
    ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        result["geom_distance"] = [
            distance_to_goal(xi, yi) for xi, yi in zip(X["x"], X["y"])
        ]
        result["geom_angle_rad"] = [
            angle_to_goal(xi, yi) for xi, yi in zip(X["x"], X["y"])
        ]
        result["geom_angle_deg"] = [
            angle_to_goal_degrees(xi, yi) for xi, yi in zip(X["x"], X["y"])
        ]
        result["geom_x_dist"] = [x_distance_to_goal_line(xi) for xi in X["x"]]
        result["geom_y_dist"] = [y_distance_to_goal_center(yi) for yi in X["y"]]
        result["geom_in_box"] = [
            int(is_in_box(xi, yi)) for xi, yi in zip(X["x"], X["y"])
        ]
        return result

    def get_feature_names_out(self, input_features=None):
        return list(input_features or []) + self.OUTPUT_COLS


class FreezeFrameFeatures(BaseEstimator, TransformerMixin):
    """Add freeze-frame derived features (defenders, GK position, etc.).

    Adds:
        ff_has_frame, ff_has_gk
        ff_n_opponents_in_cone
        ff_n_opponents_within_3m, ff_n_opponents_within_5m
        ff_dist_nearest_opponent
        ff_dist_to_gk
        ff_gk_off_line, ff_gk_y_offset
        ff_n_teammates_in_box

    Requires input columns: x, y, shot_freeze_frame
    """

    OUTPUT_COLS = [
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
    ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        records = [
            compute_all_freeze_frame_features(raw, xi, yi)
            for raw, xi, yi in zip(
                X["shot_freeze_frame"], X["x"], X["y"]
            )
        ]
        ff_df = pd.DataFrame(records, index=X.index)
        for col in self.OUTPUT_COLS:
            result[col] = ff_df[col]
        return result

    def get_feature_names_out(self, input_features=None):
        return list(input_features or []) + self.OUTPUT_COLS
