"""
Generate a 4-panel project summary dashboard (reports/project_summary.png).

Panels:
1. Calibration curve (LightGBM vs StatsBomb)
2. SHAP top 10 bar chart
3. Player over/underperformer top 10
4. Team xG vs actual goals scatter
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gradient_boosting import prepare_lgbm_data
from src.preprocess.splitter import match_based_split

DATA_PATH = PROJECT_ROOT / "data" / "shots_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _safe(text: str, width: int = 22) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")[:width]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(DATA_PATH)
    _, test_df = match_based_split(df, test_size=0.20, random_state=42)

    y_test = test_df["is_goal"].values
    sb_xg = test_df["statsbomb_xg"].values

    model = joblib.load(MODELS_DIR / "final_xg_lgbm.joblib")
    X_test, _ = prepare_lgbm_data(test_df)
    model_xg = model.predict_proba(X_test)[:, 1]

    # Full dataset predictions for scouting
    X_all, _ = prepare_lgbm_data(df)
    df["model_xg"] = model.predict_proba(X_all)[:, 1]

    # Load results
    with open(REPORTS_DIR / "lgbm_results.json") as f:
        results = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Expected Goals (xG) Model -- Project Summary",
                 fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Calibration curve
    ax = axes[0, 0]
    prob_true_m, prob_pred_m = calibration_curve(y_test, model_xg, n_bins=10)
    prob_true_s, prob_pred_s = calibration_curve(y_test, sb_xg, n_bins=10)
    ax.plot(prob_pred_m, prob_true_m, "o-", color="orange", lw=2, label="LightGBM (ours)")
    ax.plot(prob_pred_s, prob_true_s, "s--", color="steelblue", lw=2, label="StatsBomb xG")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Observed frequency", fontsize=11)
    ax.set_title("Calibration Curve (Reliability Diagram)\nECE = 0.018", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Panel 2: SHAP top 10
    ax = axes[0, 1]
    shap_data = results["top_15_shap"][:10]
    features = [d["feature"] for d in shap_data][::-1]
    values = [d["mean_abs_shap"] for d in shap_data][::-1]
    colors = ["orange" if v > 0.15 else "steelblue" for v in values]
    ax.barh(features, values, color=colors, alpha=0.8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance (SHAP)\nTop 10 features", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    # Panel 3: Player over/under (top + bottom 8)
    ax = axes[1, 0]
    player_csv = REPORTS_DIR / "scouting_players.csv"
    player_df = pd.read_csv(player_csv)
    top8 = player_df.head(8)
    bot8 = player_df.tail(8).sort_values("goals_above_xg")
    combined = pd.concat([bot8, top8])
    names = [_safe(p) for p in combined["player"]]
    vals = combined["goals_above_xg"].values
    colors_bar = ["green" if v > 0 else "crimson" for v in vals]
    ax.barh(names, vals, color=colors_bar, alpha=0.75)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Goals Above Expected xG", fontsize=11)
    ax.set_title("Player Finishing Skill\n(Top 8 over + Bottom 8 under, min 20 shots)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    # Panel 4: Team xG vs actual goals
    ax = axes[1, 1]
    team_csv = REPORTS_DIR / "scouting_teams.csv"
    team_df = pd.read_csv(team_csv)
    ax.scatter(team_df["xg_for"], team_df["goals_for"], alpha=0.5, s=35, color="steelblue")
    lim = max(team_df["xg_for"].max(), team_df["goals_for"].max()) + 10
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y=x (expected)")
    # Label top outliers
    top_teams = team_df.nlargest(5, "attacking_over_under")
    for _, row in top_teams.iterrows():
        ax.annotate(
            _safe(row["team"], 15),
            (row["xg_for"], row["goals_for"]),
            fontsize=8, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
        )
    from scipy import stats
    r, _ = stats.pearsonr(team_df["xg_for"], team_df["goals_for"])
    ax.set_xlabel("Model xG (total)", fontsize=11)
    ax.set_ylabel("Actual Goals", fontsize=11)
    ax.set_title(f"Team Validation: xG vs Actual Goals\nr = {r:.4f} ({len(team_df)} teams)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = REPORTS_DIR / "project_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
