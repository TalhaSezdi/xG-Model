"""
Phase 5: Validation & Benchmark

Steps:
1. Load data, match-based split (same seed as training)
2. Generate model predictions on test set
3. Shot-level correlation: model_xg vs statsbomb_xg
4. Team-level aggregation: xG totals vs actual goals
5. Player-level aggregation (min 10 shots)
6. Error analysis: high-xG misses and low-xG goals
7. Save reports
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gradient_boosting import prepare_lgbm_data
from src.preprocess.splitter import match_based_split, validate_split

DATA_PATH = PROJECT_ROOT / "data" / "shots_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MIN_PLAYER_SHOTS = 10


def load_model_and_predict(model_path: Path, X: pd.DataFrame) -> np.ndarray:
    """Load saved model and predict probabilities."""
    model = joblib.load(model_path)
    return model.predict_proba(X)[:, 1]


def shot_level_correlation(
    y_true: np.ndarray,
    model_xg: np.ndarray,
    sb_xg: np.ndarray,
    output_path: Path,
) -> dict:
    """Compute and plot shot-level correlation between model and StatsBomb xG."""
    r_model_sb, p_model_sb = stats.pearsonr(model_xg, sb_xg)
    r_model_actual, _ = stats.pearsonr(model_xg, y_true.astype(float))
    r_sb_actual, _ = stats.pearsonr(sb_xg, y_true.astype(float))

    print(f"  Shot-level: model vs statsbomb_xg  -> r={r_model_sb:.4f}", flush=True)
    print(f"  Shot-level: model vs actual_goal   -> r={r_model_actual:.4f}", flush=True)
    print(f"  Shot-level: statsbomb vs actual_goal -> r={r_sb_actual:.4f}", flush=True)

    # Scatter: model_xg vs statsbomb_xg (hexbin for density)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    hb = ax.hexbin(sb_xg, model_xg, gridsize=50, cmap="Blues", mincnt=1)
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="y=x (perfect agreement)")
    plt.colorbar(hb, ax=ax, label="Shot count")
    ax.set_xlabel("StatsBomb xG", fontsize=12)
    ax.set_ylabel("Model xG", fontsize=12)
    ax.set_title(f"Shot-level: Model vs StatsBomb xG\nPearson r={r_model_sb:.4f}", fontsize=13)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Mean xG per decile of StatsBomb xG
    ax2 = axes[1]
    deciles = pd.qcut(sb_xg, q=10, labels=False, duplicates="drop")
    model_means = pd.Series(model_xg).groupby(deciles).mean().values
    sb_means = pd.Series(sb_xg).groupby(deciles).mean().values
    actual_means = pd.Series(y_true.astype(float)).groupby(deciles).mean().values

    x = np.arange(len(sb_means))
    width = 0.3
    ax2.bar(x - width, sb_means, width, label="StatsBomb xG", color="steelblue", alpha=0.8)
    ax2.bar(x, model_means, width, label="Model xG", color="orange", alpha=0.8)
    ax2.bar(x + width, actual_means, width, label="Actual goal rate", color="green", alpha=0.8)
    ax2.set_xlabel("StatsBomb xG decile (low -> high)", fontsize=12)
    ax2.set_ylabel("Mean value", fontsize=12)
    ax2.set_title("xG decile comparison\n(Model vs StatsBomb vs Actual)", fontsize=13)
    ax2.legend()
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"D{i+1}" for i in range(len(x))])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}", flush=True)

    return {
        "r_model_vs_statsbomb": float(r_model_sb),
        "r_model_vs_actual": float(r_model_actual),
        "r_statsbomb_vs_actual": float(r_sb_actual),
    }


def team_aggregation(
    test_df: pd.DataFrame,
    model_xg: np.ndarray,
    output_path: Path,
) -> dict:
    """Aggregate xG and goals per team, compute correlation with actual goals."""
    df = test_df[["team", "is_goal", "statsbomb_xg"]].copy()
    df["model_xg"] = model_xg

    team_stats = df.groupby("team").agg(
        actual_goals=("is_goal", "sum"),
        model_xg_total=("model_xg", "sum"),
        sb_xg_total=("statsbomb_xg", "sum"),
        n_shots=("is_goal", "count"),
    ).reset_index()

    # Only teams with enough shots
    team_stats = team_stats[team_stats["n_shots"] >= 20].copy()

    r_model, _ = stats.pearsonr(team_stats["model_xg_total"], team_stats["actual_goals"])
    r_sb, _ = stats.pearsonr(team_stats["sb_xg_total"], team_stats["actual_goals"])

    print(f"  Team-level: model_xg vs actual_goals  -> r={r_model:.4f} (n={len(team_stats)} teams)", flush=True)
    print(f"  Team-level: statsbomb_xg vs actual_goals -> r={r_sb:.4f}", flush=True)

    top_overperform = team_stats.nlargest(5, "actual_goals")
    print("\n  Top 5 teams by actual goals (test set):", flush=True)
    for _, row in top_overperform.iterrows():
        diff = row["actual_goals"] - row["model_xg_total"]
        print(f"    {row['team'][:30]:<30} goals={row['actual_goals']:.0f}  "
              f"model_xg={row['model_xg_total']:.1f}  diff={diff:+.1f}", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    ax1.scatter(team_stats["model_xg_total"], team_stats["actual_goals"],
                alpha=0.6, s=40, color="orange", label=f"Model xG (r={r_model:.3f})")
    ax1.scatter(team_stats["sb_xg_total"], team_stats["actual_goals"],
                alpha=0.4, s=40, color="steelblue", label=f"StatsBomb xG (r={r_sb:.3f})", marker="^")
    # Best fit line for model
    m, b = np.polyfit(team_stats["model_xg_total"], team_stats["actual_goals"], 1)
    xline = np.linspace(team_stats["model_xg_total"].min(), team_stats["model_xg_total"].max(), 100)
    ax1.plot(xline, m * xline + b, "orange", lw=1.5, alpha=0.7)
    # y=x reference
    lims = [0, max(team_stats["actual_goals"].max(), team_stats["model_xg_total"].max()) + 2]
    ax1.plot(lims, lims, "k--", lw=1, alpha=0.4, label="y=x")
    ax1.set_xlabel("Expected Goals (xG total)", fontsize=12)
    ax1.set_ylabel("Actual Goals", fontsize=12)
    ax1.set_title(f"Team-level: xG vs Actual Goals\n(n={len(team_stats)} teams, test set)", fontsize=13)
    ax1.legend()

    # Over/under performance distribution
    ax2 = axes[1]
    team_stats["over_under"] = team_stats["actual_goals"] - team_stats["model_xg_total"]
    ax2.hist(team_stats["over_under"], bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", lw=2, linestyle="--", label="Zero (expected)")
    ax2.axvline(team_stats["over_under"].mean(), color="orange", lw=1.5,
                linestyle="--", label=f"Mean={team_stats['over_under'].mean():.2f}")
    ax2.set_xlabel("Actual Goals - Model xG", fontsize=12)
    ax2.set_ylabel("Number of teams", fontsize=12)
    ax2.set_title("Team Over/Under-Performance Distribution\n(positive = scored more than expected)", fontsize=13)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}", flush=True)

    return {
        "n_teams": int(len(team_stats)),
        "r_model_vs_actual": float(r_model),
        "r_statsbomb_vs_actual": float(r_sb),
        "mean_residual": float(team_stats["over_under"].mean()),
        "std_residual": float(team_stats["over_under"].std()),
    }


def _safe(text: str, width: int = 28) -> str:
    """Encode string to ASCII, replacing non-ASCII chars, then pad/truncate."""
    return text.encode("ascii", errors="replace").decode("ascii")[:width]


def player_aggregation(
    test_df: pd.DataFrame,
    model_xg: np.ndarray,
    min_shots: int = MIN_PLAYER_SHOTS,
) -> dict:
    """Aggregate xG per player, compute correlation."""
    df = test_df[["player", "is_goal", "statsbomb_xg"]].copy()
    df["model_xg"] = model_xg

    player_stats = df.groupby("player").agg(
        actual_goals=("is_goal", "sum"),
        model_xg_total=("model_xg", "sum"),
        sb_xg_total=("statsbomb_xg", "sum"),
        n_shots=("is_goal", "count"),
    ).reset_index()

    player_stats = player_stats[player_stats["n_shots"] >= min_shots].copy()
    player_stats["over_under"] = player_stats["actual_goals"] - player_stats["model_xg_total"]

    r_model, _ = stats.pearsonr(player_stats["model_xg_total"], player_stats["actual_goals"])
    r_sb, _ = stats.pearsonr(player_stats["sb_xg_total"], player_stats["actual_goals"])

    print(f"  Player-level: model_xg vs actual_goals  -> r={r_model:.4f} (n={len(player_stats)} players)", flush=True)
    print(f"  Player-level: statsbomb_xg vs actual_goals -> r={r_sb:.4f}", flush=True)

    top5 = player_stats.nlargest(5, "over_under")
    bot5 = player_stats.nsmallest(5, "over_under")
    print(f"\n  Top 5 overperformers (actual - model_xg):", flush=True)
    for _, row in top5.iterrows():
        print(f"    {_safe(row['player']):<28} goals={row['actual_goals']:.0f}  "
              f"xG={row['model_xg_total']:.2f}  diff={row['over_under']:+.2f}  shots={row['n_shots']:.0f}", flush=True)
    print(f"\n  Top 5 underperformers (actual - model_xg):", flush=True)
    for _, row in bot5.iterrows():
        print(f"    {_safe(row['player']):<28} goals={row['actual_goals']:.0f}  "
              f"xG={row['model_xg_total']:.2f}  diff={row['over_under']:+.2f}  shots={row['n_shots']:.0f}", flush=True)

    return {
        "n_players": int(len(player_stats)),
        "min_shots_threshold": min_shots,
        "r_model_vs_actual": float(r_model),
        "r_statsbomb_vs_actual": float(r_sb),
        "top_overperformers": top5[["player", "actual_goals", "model_xg_total", "over_under", "n_shots"]].to_dict("records"),
        "top_underperformers": bot5[["player", "actual_goals", "model_xg_total", "over_under", "n_shots"]].to_dict("records"),
    }


def error_analysis(
    test_df: pd.DataFrame,
    model_xg: np.ndarray,
    high_xg_thresh: float = 0.5,
    low_xg_thresh: float = 0.1,
) -> dict:
    """Analyze high-xG misses and low-xG goals."""
    df = test_df.copy()
    df["model_xg"] = model_xg

    high_miss = df[(df["model_xg"] >= high_xg_thresh) & (df["is_goal"] == 0)]
    low_goal = df[(df["model_xg"] <= low_xg_thresh) & (df["is_goal"] == 1)]
    correct_high = df[(df["model_xg"] >= high_xg_thresh) & (df["is_goal"] == 1)]

    print(f"\n  High xG (>={high_xg_thresh}) misses: {len(high_miss):,}", flush=True)
    print(f"  High xG (>={high_xg_thresh}) goals:  {len(correct_high):,}", flush=True)
    print(f"  Conversion rate at high xG: {len(correct_high) / (len(high_miss) + len(correct_high)):.1%}", flush=True)
    print(f"\n  Low xG (<={low_xg_thresh}) goals:  {len(low_goal):,}", flush=True)
    print(f"  Low xG shots total:  {len(df[df['model_xg'] <= low_xg_thresh]):,}", flush=True)

    print("\n  High-xG misses -- avg features:", flush=True)
    numeric_cols = ["geom_distance", "geom_angle_deg", "ff_dist_to_gk", "ff_n_opponents_in_cone"]
    for col in numeric_cols:
        if col in df.columns:
            print(f"    {col}: miss={high_miss[col].mean():.2f}  goal={correct_high[col].mean():.2f}", flush=True)

    print("\n  Shot type breakdown in high-xG misses:", flush=True)
    if "shot_body_part" in df.columns:
        bp = high_miss["shot_body_part"].value_counts()
        for k, v in bp.items():
            print(f"    {k}: {v} ({v/len(high_miss):.1%})", flush=True)

    print("\n  Low-xG goals -- avg features:", flush=True)
    for col in numeric_cols:
        if col in df.columns:
            low_goal_mean = low_goal[col].mean()
            all_goals_mean = df[df["is_goal"] == 1][col].mean()
            print(f"    {col}: low_xg_goals={low_goal_mean:.2f}  all_goals={all_goals_mean:.2f}", flush=True)

    print("\n  Low-xG goal patterns:", flush=True)
    if "shot_deflected" in df.columns:
        deflected_pct = low_goal["shot_deflected"].mean()
        print(f"    Deflected: {deflected_pct:.1%} (vs {df[df['is_goal']==1]['shot_deflected'].mean():.1%} all goals)", flush=True)
    if "shot_open_goal" in df.columns:
        og_pct = low_goal["shot_open_goal"].mean()
        print(f"    Open goal: {og_pct:.1%}", flush=True)

    return {
        "high_xg_miss_count": int(len(high_miss)),
        "high_xg_goal_count": int(len(correct_high)),
        "high_xg_conversion_rate": float(len(correct_high) / max(1, len(high_miss) + len(correct_high))),
        "low_xg_goal_count": int(len(low_goal)),
        "high_miss_avg_distance": float(high_miss["geom_distance"].mean()) if "geom_distance" in df.columns else None,
        "low_goal_avg_distance": float(low_goal["geom_distance"].mean()) if "geom_distance" in df.columns else None,
        "low_goal_deflected_rate": float(low_goal["shot_deflected"].mean()) if "shot_deflected" in df.columns else None,
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and split
    print("=" * 60, flush=True)
    print("STEP 1: Load data and split", flush=True)
    print("=" * 60, flush=True)
    df = pd.read_parquet(DATA_PATH)
    _, test_df = match_based_split(df, test_size=0.20, random_state=42)
    print(f"  Test set: {len(test_df):,} shots", flush=True)

    y_test = test_df["is_goal"].values
    sb_xg = test_df["statsbomb_xg"].values

    # 2. Generate predictions
    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Load model and predict", flush=True)
    print("=" * 60, flush=True)
    model_path = MODELS_DIR / "final_xg_lgbm.joblib"
    X_test, _ = prepare_lgbm_data(test_df)
    model_xg = load_model_and_predict(model_path, X_test)
    print(f"  Predictions: min={model_xg.min():.4f}, max={model_xg.max():.4f}, "
          f"mean={model_xg.mean():.4f}", flush=True)

    # 3. Shot-level correlation
    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Shot-level correlation", flush=True)
    print("=" * 60, flush=True)
    shot_corr = shot_level_correlation(
        y_test, model_xg, sb_xg,
        output_path=REPORTS_DIR / "shot_correlation.png",
    )

    # 4. Team aggregation
    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Team-level aggregation", flush=True)
    print("=" * 60, flush=True)
    team_results = team_aggregation(
        test_df, model_xg,
        output_path=REPORTS_DIR / "team_aggregation.png",
    )

    # 5. Player aggregation
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Player-level aggregation", flush=True)
    print("=" * 60, flush=True)
    player_results = player_aggregation(test_df, model_xg, min_shots=MIN_PLAYER_SHOTS)

    # 6. Error analysis
    print("\n" + "=" * 60, flush=True)
    print("STEP 6: Error analysis", flush=True)
    print("=" * 60, flush=True)
    error_results = error_analysis(test_df, model_xg)

    # 7. Save results
    print("\n" + "=" * 60, flush=True)
    print("STEP 7: Save results", flush=True)
    print("=" * 60, flush=True)
    results = {
        "shot_level_correlation": shot_corr,
        "team_aggregation": team_results,
        "player_aggregation": player_results,
        "error_analysis": error_results,
    }
    results_path = REPORTS_DIR / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {results_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
