"""Unit tests for the geometric xG features.

These cover the two features that dominate SHAP importance (distance and angle)
plus the box check and the documented edge cases.
"""

import math

import pytest

from src.features.geometry import (
    GOAL_X,
    GOAL_Y_CENTER,
    angle_to_goal,
    angle_to_goal_degrees,
    distance_to_goal,
    is_in_box,
    x_distance_to_goal_line,
    y_distance_to_goal_center,
)


class TestDistanceToGoal:
    def test_on_goal_center_is_zero(self):
        assert distance_to_goal(GOAL_X, GOAL_Y_CENTER) == pytest.approx(0.0)

    def test_penalty_spot_is_twelve(self):
        # Penalty spot (108, 40) is directly in front of goal center.
        assert distance_to_goal(108.0, 40.0) == pytest.approx(12.0)

    def test_lateral_offset_uses_hypotenuse(self):
        # (108, 31): dx = 12, dy = 9 -> 3-4-5 triangle scaled by 3 -> 15.
        assert distance_to_goal(108.0, 31.0) == pytest.approx(15.0)


class TestAngleToGoal:
    def test_behind_goal_line_returns_zero(self):
        assert angle_to_goal(GOAL_X, 40.0) == 0.0
        assert angle_to_goal(GOAL_X + 5.0, 40.0) == 0.0

    def test_penalty_spot_known_angle(self):
        # Posts at (120, 36) and (120, 44); shot at (108, 40).
        # Vectors (12, -4) and (12, 4): cos = 128 / 160 = 0.8.
        assert angle_to_goal(108.0, 40.0) == pytest.approx(math.acos(0.8))

    def test_angle_shrinks_with_distance(self):
        # Farther straight-on shots see a narrower goal.
        near = angle_to_goal(110.0, 40.0)
        far = angle_to_goal(60.0, 40.0)
        assert near > far

    def test_angle_in_valid_range(self):
        for x, y in [(108.0, 40.0), (100.0, 20.0), (90.0, 60.0)]:
            assert 0.0 <= angle_to_goal(x, y) <= math.pi

    def test_degrees_matches_radians(self):
        rad = angle_to_goal(108.0, 40.0)
        assert angle_to_goal_degrees(108.0, 40.0) == pytest.approx(math.degrees(rad))


class TestDistanceHelpers:
    def test_x_distance_clamped_at_goal_line(self):
        assert x_distance_to_goal_line(100.0) == pytest.approx(20.0)
        assert x_distance_to_goal_line(GOAL_X + 5.0) == 0.0

    def test_y_distance_is_absolute(self):
        assert y_distance_to_goal_center(40.0) == pytest.approx(0.0)
        assert y_distance_to_goal_center(20.0) == pytest.approx(20.0)
        assert y_distance_to_goal_center(60.0) == pytest.approx(20.0)


class TestIsInBox:
    def test_inside_box(self):
        assert is_in_box(108.0, 40.0) is True

    def test_outside_box_too_far(self):
        assert is_in_box(60.0, 40.0) is False

    def test_outside_box_too_wide(self):
        assert is_in_box(110.0, 10.0) is False

    def test_box_boundaries_inclusive(self):
        assert is_in_box(102.0, 18.0) is True
        assert is_in_box(120.0, 62.0) is True
        assert is_in_box(101.9, 40.0) is False
