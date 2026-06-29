"""
Train the baseline Logistic Regression xG model.

Steps:
1. Load shots_features.parquet
2. Match-based train/test split
3. Fit baseline pipeline (imputer + OHE + LR) on train
4. Evaluate on test: log-loss, Brier, AUC, calibration curve
5. Compare against StatsBomb xG on same test set
6. Print LR coefficients for interpretability
7. Save model to models/
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_metrics,
    plot_probability_distribution,
    plot_reliability_diagram,
    print_metrics_comparison,
)
from src.models.baseline import build_baseline_pipeline, get_lr_coefficients
from src.preprocess.splitter import match_based_split, validate_split

DATA_PATH = PROJECT_ROOT / "data" / "shots_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("=" * 60, flush=True)
    print("STEP 1: Loading data", flush=True)
    print("=" * 60, flush=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} shots, {df.shape[1]} columns.", flush=True)

    # 2. Split
    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Match-based train/test split", flush=True)
    print("=" * 60, flush=True)
    train_df, test_df = match_based_split(df, test_size=0.20, random_state=42)
    split_info = validate_split(train_df, test_df)
    for k, v in split_info.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)

    # Extract target
    y_train = train_df["is_goal"].values
    y_test = test_df["is_goal"].values
    sb_xg_test = test_df["statsbomb_xg"].values

    # 3. Train
    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Training Logistic Regression baseline", flush=True)
    print("=" * 60, flush=True)
    pipeline = build_baseline_pipeline()
    pipeline.fit(train_df, y_train)
    print("  Model fitted.", flush=True)

    # 4. Predict
    y_prob_train = pipeline.predict_proba(train_df)[:, 1]
    y_prob_test = pipeline.predict_proba(test_df)[:, 1]

    # 5. Evaluate
    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Evaluation", flush=True)
    print("=" * 60, flush=True)

    metrics_train = compute_metrics(y_train, y_prob_train)
    metrics_test = compute_metrics(y_test, y_prob_test)
    metrics_sb = compute_metrics(y_test, sb_xg_test)

    print("\n--- Train metrics ---")
    for k, v in metrics_train.items():
        print(f"  {k}: {v:.6f}", flush=True)

    print("\n--- Test metrics ---")
    for k, v in metrics_test.items():
        print(f"  {k}: {v:.6f}", flush=True)

    print_metrics_comparison(metrics_test, metrics_sb,
                            model_name="Baseline LR", benchmark_name="StatsBomb xG")

    # 6. Calibration plot
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Calibration & distribution plots", flush=True)
    print("=" * 60, flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plot_reliability_diagram(
        y_test, y_prob_test, sb_xg_test,
        model_name="Baseline LR", benchmark_name="StatsBomb xG",
        ax=axes[0],
    )

    plot_probability_distribution(y_test, y_prob_test, model_name="Baseline LR", ax=axes[1])

    plt.tight_layout()
    fig_path = REPORTS_DIR / "baseline_calibration.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration plot to {fig_path}", flush=True)

    # 7. LR coefficients
    print("\n" + "=" * 60, flush=True)
    print("STEP 6: LR Coefficients (top 25)", flush=True)
    print("=" * 60, flush=True)

    coef_df = get_lr_coefficients(pipeline)
    print(coef_df.head(25).to_string(index=False), flush=True)

    # Sanity checks
    print("\n--- Sanity checks ---", flush=True)
    distance_coef = coef_df[coef_df["feature"].str.contains("geom_distance")]
    angle_coef = coef_df[coef_df["feature"].str.contains("geom_angle_rad")]
    open_goal_coef = coef_df[coef_df["feature"].str.contains("shot_open_goal")]

    if not distance_coef.empty:
        val = distance_coef.iloc[0]["coefficient"]
        status = "PASS" if val < 0 else "FAIL"
        print(f"  geom_distance coefficient: {val:.4f} ({status} -- should be negative)", flush=True)
    if not angle_coef.empty:
        val = angle_coef.iloc[0]["coefficient"]
        status = "PASS" if val > 0 else "FAIL"
        print(f"  geom_angle_rad coefficient: {val:.4f} ({status} -- should be positive)", flush=True)
    if not open_goal_coef.empty:
        val = open_goal_coef.iloc[0]["coefficient"]
        status = "PASS" if val > 0 else "FAIL"
        print(f"  shot_open_goal coefficient: {val:.4f} ({status} -- should be positive)", flush=True)

    # 8. Save model
    model_path = MODELS_DIR / "baseline_lr.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n  Model saved to {model_path}", flush=True)

    # Save metrics
    results = {
        "model": "Baseline LR",
        "train": metrics_train,
        "test": metrics_test,
        "benchmark_statsbomb": metrics_sb,
        "split": split_info,
    }
    results_path = REPORTS_DIR / "baseline_results.json"
    import json
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
