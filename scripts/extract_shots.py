"""
Extract shot events from StatsBomb Open Data.

Uses StatsBomb GitHub raw JSON for speed instead of the slow sb.events() API.
Pulls all male professional competitions, processes the full event stream of
each match to enrich shots with:
  - native StatsBomb shot flags (open_goal, one_on_one, first_time, ...)
  - key-pass (assist) characteristics via shot.key_pass_id lookup
  - game state (running score difference before the shot)

Applies scope filters (no penalties, no own goals) and saves the result as
data/shots_raw.parquet.
"""

import json
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from statsbombpy import sb

warnings.filterwarnings("ignore", category=UserWarning, message="credentials were not supplied")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

STATSBOMB_EVENTS_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
)


def get_male_competitions() -> pd.DataFrame:
    comps = sb.competitions()
    male_comps = comps[comps["competition_gender"] == "male"]
    print(f"Found {len(male_comps)} male competition-seasons.", flush=True)
    return male_comps


def get_all_matches(competitions: pd.DataFrame) -> pd.DataFrame:
    all_matches = []
    for _, row in competitions.iterrows():
        comp_id = row["competition_id"]
        season_id = row["season_id"]
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            matches["competition_name"] = row["competition_name"]
            matches["season_name"] = row["season_name"]
            all_matches.append(matches)
        except Exception as e:
            print(f"  SKIP comp={comp_id} season={season_id}: {e}", flush=True)
    result = pd.concat(all_matches, ignore_index=True)
    print(f"Total matches: {len(result)}", flush=True)
    return result


def _flag(shot: dict, key: str) -> int:
    """StatsBomb only writes boolean shot flags when True; absent => False."""
    return int(bool(shot.get(key, False)))


def parse_key_pass(pass_event: Optional[dict]) -> dict:
    """Extract assist-pass characteristics from the key-pass event.

    Returns a dict with kp_* fields. If the pass event is missing, returns
    has_key_pass=0 with None values.
    """
    if pass_event is None:
        return {
            "kp_has_key_pass": 0,
            "kp_pass_height": None,
            "kp_is_cross": 0,
            "kp_is_cutback": 0,
            "kp_is_through_ball": 0,
            "kp_pass_length": None,
            "kp_pass_angle": None,
        }

    p = pass_event.get("pass", {}) or {}
    technique = (p.get("technique", {}) or {}).get("name")
    height = (p.get("height", {}) or {}).get("name")

    return {
        "kp_has_key_pass": 1,
        "kp_pass_height": height,
        "kp_is_cross": int(bool(p.get("cross", False))),
        "kp_is_cutback": int(bool(p.get("cut_back", False))),
        "kp_is_through_ball": int(technique == "Through Ball"),
        "kp_pass_length": p.get("length"),
        "kp_pass_angle": p.get("angle"),
    }


def parse_shot_event(
    event: dict,
    competition: str,
    season: str,
    score_diff: int,
    key_pass_event: Optional[dict],
) -> dict:
    """Parse a raw StatsBomb shot event JSON into a flat record."""
    location = event.get("location", [None, None])
    x = location[0] if location and len(location) > 0 else None
    y = location[1] if location and len(location) > 1 else None

    shot = event.get("shot", {}) or {}

    freeze_frame = shot.get("freeze_frame")
    if freeze_frame is not None:
        try:
            freeze_frame_str = json.dumps(freeze_frame)
        except (TypeError, ValueError):
            freeze_frame_str = None
    else:
        freeze_frame_str = None

    record = {
        "match_id": event.get("match_id"),
        "competition": competition,
        "season": season,
        "minute": event.get("minute"),
        "second": event.get("second"),
        "player": (event.get("player", {}) or {}).get("name"),
        "team": (event.get("team", {}) or {}).get("name"),
        "x": x,
        "y": y,
        "shot_outcome": (shot.get("outcome", {}) or {}).get("name"),
        "shot_type": (shot.get("type", {}) or {}).get("name"),
        "shot_body_part": (shot.get("body_part", {}) or {}).get("name"),
        "shot_technique": (shot.get("technique", {}) or {}).get("name"),
        "play_pattern": (event.get("play_pattern", {}) or {}).get("name"),
        "under_pressure": bool(event.get("under_pressure", False)),
        "shot_freeze_frame": freeze_frame_str,
        "statsbomb_xg": shot.get("statsbomb_xg"),
        # Native shot flags
        "shot_open_goal": _flag(shot, "open_goal"),
        "shot_one_on_one": _flag(shot, "one_on_one"),
        "shot_first_time": _flag(shot, "first_time"),
        "shot_deflected": _flag(shot, "deflected"),
        "shot_follows_dribble": _flag(shot, "follows_dribble"),
        "shot_aerial_won": _flag(shot, "aerial_won"),
        "shot_redirect": _flag(shot, "redirect"),
        # Game state
        "score_diff": score_diff,
    }
    record.update(parse_key_pass(key_pass_event))
    return record


def process_match(match_id: int, competition: str, season: str) -> list[dict]:
    """Fetch full event stream, compute running score, enrich and return shots."""
    url = STATSBOMB_EVENTS_URL.format(match_id=match_id)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        events = resp.json()
    except Exception:
        return []

    # Chronological order (StatsBomb 'index' is the canonical sequence).
    events = sorted(events, key=lambda e: e.get("index", 0))

    # Lookup table for key-pass resolution.
    events_by_id = {e.get("id"): e for e in events}

    score: dict[str, int] = defaultdict(int)
    records: list[dict] = []

    for event in events:
        etype = (event.get("type", {}) or {}).get("name")
        team = (event.get("team", {}) or {}).get("name")

        if etype == "Shot":
            shot = event.get("shot", {}) or {}
            # Snapshot score BEFORE this shot's own outcome is applied.
            opponent_goals = sum(g for t, g in score.items() if t != team)
            score_diff = score.get(team, 0) - opponent_goals

            key_pass_id = shot.get("key_pass_id")
            key_pass_event = events_by_id.get(key_pass_id) if key_pass_id else None

            event["match_id"] = match_id
            records.append(
                parse_shot_event(event, competition, season, score_diff, key_pass_event)
            )

            # Apply goal to running score after snapshot.
            if (shot.get("outcome", {}) or {}).get("name") == "Goal":
                score[team] += 1

        elif etype == "Own Goal For":
            # Team in this event benefits from the own goal.
            if team is not None:
                score[team] += 1

    return records


def apply_scope_filters(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)

    n_penalties = (df["shot_type"] == "Penalty").sum()
    df = df[df["shot_type"] != "Penalty"].copy()
    print(f"Removed {n_penalties} penalties.", flush=True)

    own_goal_mask = df["shot_outcome"].str.contains("Own Goal", case=False, na=False)
    n_own_goals = own_goal_mask.sum()
    df = df[~own_goal_mask].copy()
    print(f"Removed {n_own_goals} own goals.", flush=True)

    print(f"Shots: {n_before} -> {len(df)} after filters.", flush=True)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["is_goal"] = (df["shot_outcome"] == "Goal").astype(int)
    goal_rate = df["is_goal"].mean()
    print(f"Goal rate: {goal_rate:.4f} ({df['is_goal'].sum()} goals / {len(df)} shots)", flush=True)
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("STEP 1: Fetching male competitions", flush=True)
    print("=" * 60, flush=True)
    competitions = get_male_competitions()

    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Fetching all matches", flush=True)
    print("=" * 60, flush=True)
    matches = get_all_matches(competitions)

    match_meta = {}
    for _, row in matches.iterrows():
        mid = row["match_id"]
        if mid not in match_meta:
            match_meta[mid] = (row.get("competition_name", ""), row.get("season_name", ""))

    match_ids = list(match_meta.keys())
    total = len(match_ids)

    print(f"\n{'=' * 60}", flush=True)
    print(f"STEP 3: Processing {total} matches (parallel, full event stream)", flush=True)
    print("=" * 60, flush=True)

    all_records = []
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_mid = {
            executor.submit(process_match, mid, match_meta[mid][0], match_meta[mid][1]): mid
            for mid in match_ids
        }

        for future in as_completed(future_to_mid):
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  Progress: {done}/{total} matches processed...", flush=True)
            try:
                records = future.result()
            except Exception:
                errors += 1
                continue
            all_records.extend(records)

    print(f"  Total raw shot records: {len(all_records)}", flush=True)
    print(f"  Errors: {errors}", flush=True)

    df = pd.DataFrame(all_records)

    print(f"\n{'=' * 60}", flush=True)
    print("STEP 4: Applying scope filters", flush=True)
    print("=" * 60, flush=True)
    df = apply_scope_filters(df)

    print(f"\n{'=' * 60}", flush=True)
    print("STEP 5: Creating target variable", flush=True)
    print("=" * 60, flush=True)
    df = create_target(df)

    print(f"\n{'=' * 60}", flush=True)
    print("STEP 6: Validation", flush=True)
    print("=" * 60, flush=True)
    print(f"Shape: {df.shape}", flush=True)

    new_cols = [
        "shot_open_goal", "shot_one_on_one", "shot_first_time", "shot_deflected",
        "shot_follows_dribble", "shot_aerial_won", "shot_redirect",
        "kp_has_key_pass", "kp_pass_height", "kp_is_cross", "kp_is_cutback",
        "kp_is_through_ball", "kp_pass_length", "kp_pass_angle", "score_diff",
    ]
    print("\nNew feature summary:", flush=True)
    for c in new_cols:
        if c in ("kp_pass_height",):
            print(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}", flush=True)
        elif c in ("kp_pass_length", "kp_pass_angle", "score_diff"):
            print(f"  {c}: mean={df[c].mean():.3f} nulls={df[c].isnull().sum()}", flush=True)
        else:
            print(f"  {c}: sum={df[c].sum()} ({df[c].mean()*100:.1f}% positive)", flush=True)

    output_path = DATA_DIR / "shots_raw.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}", flush=True)
    print(f"Final shape: {df.shape}", flush=True)


if __name__ == "__main__":
    main()
