"""
Geometric features for xG modeling.

StatsBomb pitch coordinates:
    x = [0, 120] (length)
    y = [0, 80]  (width)

Goal is at x = 120, centered at y = 40.
Goal width = 8 yards ~ 7.32m, so goal posts at y = 36 and y = 44.

All functions in this module are stateless (no fitting required).
"""

from __future__ import annotations

import math

GOAL_X = 120.0
GOAL_Y_CENTER = 40.0
GOAL_Y_LEFT = 36.0
GOAL_Y_RIGHT = 44.0
GOAL_WIDTH = GOAL_Y_RIGHT - GOAL_Y_LEFT


def distance_to_goal(x: float, y: float) -> float:
    """Euclidean distance from shot location to the goal center.

    Args:
        x: Shot x-coordinate.
        y: Shot y-coordinate.

    Returns:
        Distance in meters.
    """
    return math.hypot(GOAL_X - x, GOAL_Y_CENTER - y)


def angle_to_goal(x: float, y: float) -> float:
    """Angle (radians) subtended by the two goal posts from the shot location.

    Uses the law of cosines between two vectors: one from the shot location
    to the left post, and one to the right post. The angle between them is
    the visual width of the goal as seen by the shooter.

    Edge case: if the shooter is on or behind the goal line at goal_x,
    returns 0 (no visible goal).

    Args:
        x: Shot x-coordinate.
        y: Shot y-coordinate.

    Returns:
        Angle in radians, in [0, pi].
    """
    if x >= GOAL_X:
        return 0.0

    left_dx = GOAL_X - x
    left_dy = GOAL_Y_LEFT - y
    right_dx = GOAL_X - x
    right_dy = GOAL_Y_RIGHT - y

    dot = left_dx * right_dx + left_dy * right_dy
    norm_left = math.hypot(left_dx, left_dy)
    norm_right = math.hypot(right_dx, right_dy)

    if norm_left == 0 or norm_right == 0:
        return 0.0

    cos_theta = dot / (norm_left * norm_right)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def angle_to_goal_degrees(x: float, y: float) -> float:
    """Same as angle_to_goal but returned in degrees."""
    return math.degrees(angle_to_goal(x, y))


def x_distance_to_goal_line(x: float) -> float:
    """Horizontal distance to the goal line (always positive).

    Args:
        x: Shot x-coordinate.

    Returns:
        Distance in meters.
    """
    return max(0.0, GOAL_X - x)


def y_distance_to_goal_center(y: float) -> float:
    """Absolute lateral offset from the goal center.

    Args:
        y: Shot y-coordinate.

    Returns:
        Distance in meters.
    """
    return abs(y - GOAL_Y_CENTER)


def is_in_box(x: float, y: float) -> bool:
    """Whether the shot location is inside the 18-yard penalty box.

    StatsBomb box dimensions (in pitch units):
        x: [102, 120]
        y: [18, 62]
    """
    return 102.0 <= x <= 120.0 and 18.0 <= y <= 62.0


if __name__ == "__main__":
    test_cases = [
        ("Penalty spot (108, 40)", 108.0, 40.0),
        ("On goal line center (120, 40)", 120.0, 40.0),
        ("Corner shot (115, 75)", 115.0, 75.0),
        ("Midfield (60, 40)", 60.0, 40.0),
        ("Left wing 18yd (108, 20)", 108.0, 20.0),
    ]
    print(f"{'Case':<35} {'Dist':>8} {'Angle(deg)':>12} {'InBox':>7}")
    print("-" * 65)
    for name, x, y in test_cases:
        d = distance_to_goal(x, y)
        a = angle_to_goal_degrees(x, y)
        b = is_in_box(x, y)
        print(f"{name:<35} {d:>8.2f} {a:>12.2f} {str(b):>7}")
