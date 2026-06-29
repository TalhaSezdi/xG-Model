"""
Freeze frame features for xG modeling.

A StatsBomb freeze_frame is a list of dicts, one per player visible at the
moment of the shot (excluding the shooter). Each entry has:
    {
        "location": [x, y],
        "player": {"id": ..., "name": ...},
        "position": {"id": ..., "name": ...},
        "teammate": bool  # True if same team as shooter
    }

The shooter is NOT included. The goalkeeper IS included (position "Goalkeeper").

All functions stateless.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from .geometry import GOAL_X, GOAL_Y_CENTER, GOAL_Y_LEFT, GOAL_Y_RIGHT


def parse_freeze_frame(raw: Optional[str]) -> list[dict]:
    """Parse a freeze_frame JSON string into a list of dicts.

    Args:
        raw: JSON string or None.

    Returns:
        List of player dicts, empty if input is missing or invalid.
    """
    if raw is None or raw == "":
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def _is_opponent(entry: dict) -> bool:
    return not bool(entry.get("teammate", False))


def _is_goalkeeper(entry: dict) -> bool:
    position = entry.get("position", {}) or {}
    return position.get("name") == "Goalkeeper"


def _location(entry: dict) -> Optional[tuple[float, float]]:
    loc = entry.get("location")
    if not loc or len(loc) < 2:
        return None
    return float(loc[0]), float(loc[1])


def _point_in_triangle(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> bool:
    """Return True if point P is inside triangle ABC (or on edge).

    Uses barycentric-style sign check via cross products. Tolerates
    degenerate triangles by treating zero-area as 'not inside'.
    """
    def sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = sign(px, py, ax, ay, bx, by)
    d2 = sign(px, py, bx, by, cx, cy)
    d3 = sign(px, py, cx, cy, ax, ay)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def num_opponents_in_cone(frame: list[dict], shot_x: float, shot_y: float) -> int:
    """Count opponents inside the triangle (shot location, left post, right post).

    This is the 'shot cone' -- opponents in this area can block the shot.
    Goalkeeper is included if in the cone.
    """
    count = 0
    for entry in frame:
        if not _is_opponent(entry):
            continue
        loc = _location(entry)
        if loc is None:
            continue
        if _point_in_triangle(
            loc[0], loc[1],
            shot_x, shot_y,
            GOAL_X, GOAL_Y_LEFT,
            GOAL_X, GOAL_Y_RIGHT,
        ):
            count += 1
    return count


def num_opponents_within_radius(
    frame: list[dict], shot_x: float, shot_y: float, radius: float
) -> int:
    """Count opponents within a radius (meters) of the shot location."""
    count = 0
    r2 = radius * radius
    for entry in frame:
        if not _is_opponent(entry):
            continue
        loc = _location(entry)
        if loc is None:
            continue
        dx = loc[0] - shot_x
        dy = loc[1] - shot_y
        if dx * dx + dy * dy <= r2:
            count += 1
    return count


def distance_to_nearest_opponent(
    frame: list[dict], shot_x: float, shot_y: float
) -> Optional[float]:
    """Euclidean distance to the closest opponent. None if no opponents in frame."""
    best = None
    for entry in frame:
        if not _is_opponent(entry):
            continue
        loc = _location(entry)
        if loc is None:
            continue
        d = math.hypot(loc[0] - shot_x, loc[1] - shot_y)
        if best is None or d < best:
            best = d
    return best


def find_goalkeeper(frame: list[dict]) -> Optional[tuple[float, float]]:
    """Return goalkeeper location, or None if not visible.

    The defending goalkeeper is the one marked as opponent and position Goalkeeper.
    """
    for entry in frame:
        if not _is_opponent(entry):
            continue
        if not _is_goalkeeper(entry):
            continue
        return _location(entry)
    return None


def gk_distance_from_goal_line(gk_loc: Optional[tuple[float, float]]) -> Optional[float]:
    """How far the keeper has come off his line (x-distance)."""
    if gk_loc is None:
        return None
    return max(0.0, GOAL_X - gk_loc[0])


def gk_y_offset_from_center(gk_loc: Optional[tuple[float, float]]) -> Optional[float]:
    """Signed y-offset of GK from goal center."""
    if gk_loc is None:
        return None
    return gk_loc[1] - GOAL_Y_CENTER


def distance_to_gk(
    gk_loc: Optional[tuple[float, float]], shot_x: float, shot_y: float
) -> Optional[float]:
    """Distance from shot location to goalkeeper."""
    if gk_loc is None:
        return None
    return math.hypot(gk_loc[0] - shot_x, gk_loc[1] - shot_y)


def num_teammates_in_box(frame: list[dict]) -> int:
    """Number of attacking teammates inside the 18-yard box."""
    count = 0
    for entry in frame:
        if _is_opponent(entry):
            continue
        loc = _location(entry)
        if loc is None:
            continue
        if 102.0 <= loc[0] <= 120.0 and 18.0 <= loc[1] <= 62.0:
            count += 1
    return count


def compute_all_freeze_frame_features(
    raw_frame: Optional[str], shot_x: float, shot_y: float
) -> dict:
    """Compute every freeze-frame feature for a single shot.

    Returns a dict ready to be merged into a DataFrame row. NaN-equivalents
    (None) are used where the freeze frame is empty or the goalkeeper
    is not visible.
    """
    frame = parse_freeze_frame(raw_frame)
    has_frame = len(frame) > 0
    gk_loc = find_goalkeeper(frame) if has_frame else None

    return {
        "ff_has_frame": int(has_frame),
        "ff_n_opponents_in_cone": num_opponents_in_cone(frame, shot_x, shot_y) if has_frame else None,
        "ff_n_opponents_within_3m": num_opponents_within_radius(frame, shot_x, shot_y, 3.0) if has_frame else None,
        "ff_n_opponents_within_5m": num_opponents_within_radius(frame, shot_x, shot_y, 5.0) if has_frame else None,
        "ff_dist_nearest_opponent": distance_to_nearest_opponent(frame, shot_x, shot_y) if has_frame else None,
        "ff_dist_to_gk": distance_to_gk(gk_loc, shot_x, shot_y),
        "ff_gk_off_line": gk_distance_from_goal_line(gk_loc),
        "ff_gk_y_offset": gk_y_offset_from_center(gk_loc),
        "ff_has_gk": int(gk_loc is not None),
        "ff_n_teammates_in_box": num_teammates_in_box(frame) if has_frame else None,
    }


if __name__ == "__main__":
    sample = json.dumps([
        {"location": [110.0, 40.0], "teammate": False,
         "position": {"name": "Center Back"}},
        {"location": [119.0, 40.0], "teammate": False,
         "position": {"name": "Goalkeeper"}},
        {"location": [105.0, 35.0], "teammate": True,
         "position": {"name": "Right Wing"}},
    ])
    out = compute_all_freeze_frame_features(sample, shot_x=100.0, shot_y=40.0)
    print("Test shot at (100, 40) with 1 defender in front, GK on line, 1 teammate:")
    for k, v in out.items():
        print(f"  {k}: {v}")
