"""
Phase 6: Player Scouting Framework

Run on FULL dataset (train+test combined) for maximum sample size.
Produces player and team tables with over/under-performance rankings.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gradient_boosting import prepare_lgbm_data
from src.scouting.player_analysis import build_player_table
from src.scouting.team_analysis import build_team_table

DATA_PATH = PROJECT_ROOT / "data" / "shots_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MIN_PLAYER_SHOTS = 20
MIN_TEAM_SHOTS = 30
TOP_N = 15


def _safe(text: str, width: int = 25) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")[:width]


def plot_player_ranking(
    player_df: pd.DataFrame,
    top_n: int,
    output_path: Path,
) -> None:
    """Bar chart: top N over and under performers."""
    overperformers = player_df.head(top_n).copy()
    underperformers = player_df.tail(top_n).copy().sort_values("goals_above_xg")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, subset, title, color in [
        (axes[0], overperformers, f"Top {top_n} Overperformers\n(Goals Above xG)", "green"),
        (axes[1], underperformers, f"Top {top_n} Underperformers\n(Goals Below xG)", "crimson"),
    ]:
        names = [_safe(p) for p in subset["player"]]
        values = subset["goals_above_xg"].values
        errors = subset["goals_above_xg_se"].values
        shots = subset["shots"].values

        bars = ax.barh(names, values, xerr=errors, color=color, alpha=0.75,
                       error_kw={"elinewidth": 1.5, "capsize": 3, "ecolor": "black"})
        ax.axvline(0, color="black", lw=1.2)

        # Annotate shots
        for i, (v, s) in enumerate(zip(values, shots)):
            offset = 0.15 if v >= 0 else -0.15
            ax.text(v + offset, i, f"n={s:.0f}", va="center", fontsize=7.5, color="dimgray")

        ax.set_xlabel("Goals Above Expected xG", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Player Finishing Performance vs Model xG\n(All data, min 20 shots)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}", flush=True)


def plot_team_scatter(
    team_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter: team xG_for vs goals_for + defensive view."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Attacking
    ax = axes[0]
    ax.scatter(team_df["xg_for"], team_df["goals_for"], alpha=0.6, s=50, color="steelblue")
    lim_max = max(team_df["xg_for"].max(), team_df["goals_for"].max()) + 5
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, alpha=0.5, label="y=x (expected)")

    # Label top 8 over and under
    top_attack = pd.concat([
        team_df.nlargest(8, "attacking_over_under"),
        team_df.nsmallest(5, "attacking_over_under"),
    ]).drop_duplicates()
    for _, row in top_attack.iterrows():
        ax.annotate(
            _safe(row["team"], 18),
            (row["xg_for"], row["goals_for"]),
            fontsize=7, ha="left", va="bottom",
            xytext=(3, 3), textcoords="offset points",
        )

    ax.set_xlabel("Expected Goals For (xG)", fontsize=11)
    ax.set_ylabel("Actual Goals For", fontsize=11)
    ax.set_title("Team Attacking: xG vs Actual Goals\n(positive gap = clinical attack)", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # Over/under distribution -- attacking + defensive on same axis
    ax2 = axes[1]
    x = np.arange(len(team_df.head(20)))
    width = 0.4
    top20 = team_df.head(20).copy()
    names = [_safe(t, 20) for t in top20["team"]]

    ax2.bar(x - width / 2, top20["attacking_over_under"], width,
            label="Attacking O/U (goals_for - xg_for)", color="steelblue", alpha=0.8)
    ax2.bar(x + width / 2, top20["defensive_over_under"], width,
            label="Defensive O/U (xg_against - goals_against)", color="orange", alpha=0.8)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Goals vs Expected", fontsize=11)
    ax2.set_title("Top 20 Teams: Attacking & Defensive Over/Under\n(positive = better than expected)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}", flush=True)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load full data
    print("=" * 60, flush=True)
    print("STEP 1: Load full dataset", flush=True)
    print("=" * 60, flush=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"  Total shots: {len(df):,}", flush=True)

    # 2. Generate predictions on full dataset
    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Generate model predictions", flush=True)
    print("=" * 60, flush=True)
    model = joblib.load(MODELS_DIR / "final_xg_lgbm.joblib")
    X_all, _ = prepare_lgbm_data(df)
    df["model_xg"] = model.predict_proba(X_all)[:, 1]
    print(f"  Mean model_xg: {df['model_xg'].mean():.4f}  "
          f"(mean is_goal: {df['is_goal'].mean():.4f})", flush=True)

    # 3. Player analysis
    print("\n" + "=" * 60, flush=True)
    print(f"STEP 3: Player analysis (min {MIN_PLAYER_SHOTS} shots)", flush=True)
    print("=" * 60, flush=True)
    player_df = build_player_table(df, min_shots=MIN_PLAYER_SHOTS)
    print(f"  Players qualifying: {len(player_df)}", flush=True)

    print("\n  Top 15 overperformers:", flush=True)
    print(f"  {'Player':<28} {'Shots':>6} {'Goals':>6} {'xG':>7} {'G-xG':>7} {'SE':>6}", flush=True)
    print("  " + "-" * 65, flush=True)
    for _, row in player_df.head(15).iterrows():
        print(f"  {_safe(row['player']):<28} {row['shots']:>6.0f} "
              f"{row['actual_goals']:>6.0f} {row['xg_sum']:>7.2f} "
              f"{row['goals_above_xg']:>+7.2f} {row['goals_above_xg_se']:>6.2f}", flush=True)

    print(f"\n  Bottom 15 underperformers:", flush=True)
    print(f"  {'Player':<28} {'Shots':>6} {'Goals':>6} {'xG':>7} {'G-xG':>7} {'SE':>6}", flush=True)
    print("  " + "-" * 65, flush=True)
    for _, row in player_df.tail(15).iterrows():
        print(f"  {_safe(row['player']):<28} {row['shots']:>6.0f} "
              f"{row['actual_goals']:>6.0f} {row['xg_sum']:>7.2f} "
              f"{row['goals_above_xg']:>+7.2f} {row['goals_above_xg_se']:>6.2f}", flush=True)

    # 4. Team analysis
    print("\n" + "=" * 60, flush=True)
    print(f"STEP 4: Team analysis (min {MIN_TEAM_SHOTS} shots)", flush=True)
    print("=" * 60, flush=True)
    team_df = build_team_table(df, min_shots=MIN_TEAM_SHOTS)
    print(f"  Teams qualifying: {len(team_df)}", flush=True)

    print("\n  Top 10 attacking overperformers:", flush=True)
    print(f"  {'Team':<30} {'GF':>4} {'xGF':>7} {'Atk O/U':>8} {'Def O/U':>8}", flush=True)
    print("  " + "-" * 60, flush=True)
    for _, row in team_df.head(10).iterrows():
        print(f"  {_safe(row['team'], 30):<30} {row['goals_for']:>4.0f} "
              f"{row['xg_for']:>7.1f} {row['attacking_over_under']:>+8.1f} "
              f"{row['defensive_over_under']:>+8.1f}", flush=True)

    # 5. Plots
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Generate plots", flush=True)
    print("=" * 60, flush=True)
    plot_player_ranking(player_df, TOP_N, REPORTS_DIR / "scouting_top_players.png")
    plot_team_scatter(team_df, REPORTS_DIR / "scouting_team_scatter.png")

    # 6. Save CSVs
    print("\n" + "=" * 60, flush=True)
    print("STEP 6: Save CSV reports", flush=True)
    print("=" * 60, flush=True)
    player_csv = REPORTS_DIR / "scouting_players.csv"
    team_csv = REPORTS_DIR / "scouting_teams.csv"
    player_df.to_csv(player_csv, index=False)
    team_df.to_csv(team_csv, index=False)
    print(f"  Saved: {player_csv}", flush=True)
    print(f"  Saved: {team_csv}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
