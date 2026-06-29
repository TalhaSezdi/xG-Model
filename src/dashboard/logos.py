"""
Team logo URL loader for the dashboard.

Loads pre-fetched logo URLs from data/team_logos.json.
Falls back to a placeholder SVG data-URI when no logo exists.
"""

from __future__ import annotations

import json
from pathlib import Path

_LOGOS: dict[str, str] | None = None
_LOGOS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "team_logos.json"

# Tiny gray circle SVG as fallback (inline data-URI, no network request)
_FALLBACK = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 40 40'%3E"
    "%3Ccircle cx='20' cy='20' r='20' fill='%232A3140'/%3E"
    "%3C/svg%3E"
)


def _load() -> dict[str, str]:
    global _LOGOS
    if _LOGOS is None:
        if _LOGOS_PATH.exists():
            with open(_LOGOS_PATH, encoding="utf-8") as f:
                _LOGOS = json.load(f)
        else:
            _LOGOS = {}
    return _LOGOS


def get_logo_url(team_name: str) -> str:
    """Return a logo URL for the given team name, or a fallback SVG."""
    logos = _load()
    url = logos.get(team_name, "")
    return url if url else _FALLBACK


def add_logo_column(df, team_col: str = "team", logo_col: str = "Logo") -> None:
    """Add a logo URL column to a DataFrame in-place."""
    df.insert(0, logo_col, df[team_col].map(get_logo_url))
