"""
Team-level xG aggregation: attacking and defensive over/under-performance.

Attacking over/under = goals_for - xg_for (positive = clinical attack)
Defensive over/under = xg_against - goals_against (positive = solid defense)
"""

from __future__ import annotations

import pandas as pd


def build_team_table(
    df: pd.DataFrame,
    model_xg_col: str = "model_xg",
    target_col: str = "is_goal",
    team_col: str = "team",
    min_shots: int = 30,
) -> pd.DataFrame:
    """Aggregate per-team attacking and defensive performance.

    Each shot is from the shooting team's perspective.
    To get goals_against / xg_against, we need opponent shots.

    Strategy: for each shot, 'team' is the attacker. The shot either
    ended as a goal (is_goal=1) or not. We aggregate both perspectives:
    - Attacking: shots taken by team -> xg_for, goals_for
    - Defensive: shots faced by team (opponent shots) -> xg_against, goals_against

    This requires match_id + team info to identify opponents.
    We approximate: for each match, the two teams are the attacker team
    and the team that faced those shots. We build opponent lookup from
    match-level team pairs.

    Args:
        df: Shot-level DataFrame.
        model_xg_col: Model xG column.
        target_col: Binary goal column.
        team_col: Shooting team column.
        min_shots: Min shots (for + against combined) to include team.

    Returns:
        DataFrame with team attacking/defensive stats.
    """
    # Attacking side: shots taken by the team
    attack = df.groupby(team_col).agg(
        shots_for=(target_col, "count"),
        goals_for=(target_col, "sum"),
        xg_for=(model_xg_col, "sum"),
        sb_xg_for=("statsbomb_xg", "sum"),
    ).reset_index()
    attack.rename(columns={team_col: "team"}, inplace=True)

    # Defensive side: build match-level opponent map
    # Each row = one shot. Get all (match_id, team) combos per match.
    match_teams = (
        df.groupby(["match_id", team_col])
        .size()
        .reset_index(name="n_shots")[["match_id", team_col]]
        .drop_duplicates()
    )

    # Self-join: for each match, pair each team with the other team(s)
    opponent_map = match_teams.merge(match_teams, on="match_id", suffixes=("_shooter", "_defender"))
    opponent_map = opponent_map[opponent_map[f"{team_col}_shooter"] != opponent_map[f"{team_col}_defender"]]

    # Merge back to shots: get defensive team for each shot
    shots_with_def = df.merge(
        opponent_map[[f"match_id", f"{team_col}_shooter", f"{team_col}_defender"]],
        left_on=["match_id", team_col],
        right_on=["match_id", f"{team_col}_shooter"],
        how="left",
    )

    defense = shots_with_def.groupby(f"{team_col}_defender").agg(
        shots_against=(target_col, "count"),
        goals_against=(target_col, "sum"),
        xg_against=(model_xg_col, "sum"),
    ).reset_index()
    defense.rename(columns={f"{team_col}_defender": "team"}, inplace=True)

    # Merge attack + defense
    team_stats = attack.merge(defense, on="team", how="outer")

    # Over/under performance
    team_stats["attacking_over_under"] = team_stats["goals_for"] - team_stats["xg_for"]
    team_stats["defensive_over_under"] = team_stats["xg_against"] - team_stats["goals_against"]
    team_stats["net_xg"] = team_stats["xg_for"] - team_stats["xg_against"]
    team_stats["net_goals"] = team_stats["goals_for"] - team_stats["goals_against"]

    # Filter min shots
    total_shots = team_stats["shots_for"].fillna(0) + team_stats["shots_against"].fillna(0)
    team_stats = team_stats[total_shots >= min_shots].copy()

    team_stats = team_stats.sort_values("attacking_over_under", ascending=False).reset_index(drop=True)
    team_stats["rank"] = team_stats.index + 1

    return team_stats[
        [
            "rank", "team",
            "shots_for", "goals_for", "xg_for", "attacking_over_under",
            "shots_against", "goals_against", "xg_against", "defensive_over_under",
            "net_xg", "net_goals",
        ]
    ]
