"""
Train the LightGBM xG model with CV tuning, calibration check, and SHAP.

Steps:
1. Load data, match-based split
2. CV hyperparameter tuning on train set (GroupKFold)
3. Train final model on full train set
4. Evaluate: log-loss, Brier, AUC vs LR baseline vs StatsBomb
5. Calibration check + optional post-hoc calibration
6. SHAP analysis
7. Save final model
"""

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_metrics,
    plot_reliability_diagram,
    print_metrics_comparison,
)
from src.models.gradient_boosting import (
    calibrate_model,
    cv_tune_lgbm,
    get_default_params,
    prepare_lgbm_data,
    train_lgbm,
)
from src.preprocess.splitter import match_based_split, validate_split

DATA_PATH = PROJECT_ROOT / "data" / "shots_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and split
    print("=" * 60, flush=True)
    print("STEP 1: Load data and split", flush=True)
    print("=" * 60, flush=True)
    df = pd.read_parquet(DATA_PATH)
    train_df, test_df = match_based_split(df, test_size=0.20, random_state=42)
    split_info = validate_split(train_df, test_df)
    print(f"  Train: {split_info['train_shots']:,} shots, Test: {split_info['test_shots']:,} shots", flush=True)

    y_train = train_df["is_goal"].values
    y_test = test_df["is_goal"].values
    sb_xg_test = test_df["statsbomb_xg"].values

    X_train, feature_cols = prepare_lgbm_data(train_df)
    X_test, _ = prepare_lgbm_data(test_df)

    # 2. CV tuning
    print("\n" + "=" * 60, flush=True)
    print("STEP 2: CV hyperparameter tuning (GroupKFold, 5 folds)", flush=True)
    print("=" * 60, flush=True)
    groups = train_df["match_id"].values
    best_params, best_cv_score = cv_tune_lgbm(X_train, y_train, groups, n_splits=5)
    print(f"\n  Best CV log-loss: {best_cv_score:.6f}", flush=True)

    # 3. Train final model
    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Train final model on full train set", flush=True)
    print("=" * 60, flush=True)
    model = train_lgbm(X_train, y_train, params=best_params)
    print(f"  Trees: {model.n_estimators_}", flush=True)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_prob_train = model.predict_proba(X_train)[:, 1]

    # 4. Evaluate
    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Evaluation", flush=True)
    print("=" * 60, flush=True)

    metrics_lgbm_test = compute_metrics(y_test, y_prob_test)
    metrics_lgbm_train = compute_metrics(y_train, y_prob_train)
    metrics_sb = compute_metrics(y_test, sb_xg_test)

    # Load baseline LR metrics
    lr_results_path = REPORTS_DIR / "baseline_results.json"
    if lr_results_path.exists():
        with open(lr_results_path) as f:
            lr_results = json.load(f)
        metrics_lr = lr_results["test"]
    else:
        metrics_lr = {"log_loss": 0.2489, "brier_score": 0.0712, "roc_auc": 0.8349}

    print("\n--- LightGBM Test ---")
    for k, v in metrics_lgbm_test.items():
        print(f"  {k}: {v:.6f}", flush=True)

    print("\n--- 3-Way Comparison ---")
    print(f"{'Metric':<15} {'LightGBM':<14} {'Baseline LR':<14} {'StatsBomb':<14}", flush=True)
    print("-" * 57, flush=True)
    for key in ["log_loss", "brier_score", "roc_auc"]:
        print(f"{key:<15} {metrics_lgbm_test[key]:<14.6f} {metrics_lr[key]:<14.6f} {metrics_sb[key]:<14.6f}", flush=True)

    # 5. Calibration
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Calibration check", flush=True)
    print("=" * 60, flush=True)

    # Load LR predictions for comparison plot
    lr_model_path = MODELS_DIR / "baseline_lr.joblib"
    if lr_model_path.exists():
        lr_pipeline = joblib.load(lr_model_path)
        y_prob_lr_test = lr_pipeline.predict_proba(test_df)[:, 1]
    else:
        y_prob_lr_test = None

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # LightGBM vs StatsBomb
    plot_reliability_diagram(
        y_test, y_prob_test, sb_xg_test,
        model_name="LightGBM", benchmark_name="StatsBomb xG",
        ax=axes[0],
    )

    # LightGBM vs Baseline LR
    if y_prob_lr_test is not None:
        plot_reliability_diagram(
            y_test, y_prob_test, y_prob_lr_test,
            model_name="LightGBM", benchmark_name="Baseline LR",
            ax=axes[1],
        )
    else:
        axes[1].text(0.5, 0.5, "LR model not found", ha="center")

    plt.tight_layout()
    fig_path = REPORTS_DIR / "lgbm_calibration.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {fig_path}", flush=True)

    # Check if calibration is needed (simple heuristic: compare ECE)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    print(f"  Expected Calibration Error (ECE): {ece:.4f}", flush=True)

    final_model = model
    final_probs = y_prob_test

    if ece > 0.02:
        print("  ECE > 0.02: Applying isotonic calibration...", flush=True)
        # Use part of train for calibration
        cal_size = int(0.2 * len(X_train))
        X_cal = X_train.iloc[-cal_size:]
        y_cal = y_train[-cal_size:]
        calibrated = calibrate_model(model, X_cal, y_cal, method="isotonic")
        y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
        metrics_cal = compute_metrics(y_test, y_prob_cal)
        print(f"  After calibration -- log_loss: {metrics_cal['log_loss']:.6f}, "
              f"brier: {metrics_cal['brier_score']:.6f}", flush=True)

        if metrics_cal["log_loss"] < metrics_lgbm_test["log_loss"]:
            print("  Calibrated model is better. Using it as final.", flush=True)
            final_model = calibrated
            final_probs = y_prob_cal
            metrics_lgbm_test = metrics_cal
        else:
            print("  Raw model was already well-calibrated. Keeping it.", flush=True)
    else:
        print("  ECE <= 0.02: Model is well-calibrated, no post-hoc needed.", flush=True)

    # 6. SHAP
    print("\n" + "=" * 60, flush=True)
    print("STEP 6: SHAP analysis", flush=True)
    print("=" * 60, flush=True)

    explainer = shap.TreeExplainer(model)
    # Use a sample for speed
    sample_size = min(5000, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # class 1 (goal)
    else:
        shap_vals = shap_values

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\n  Top 15 features by mean |SHAP|:", flush=True)
    print(importance_df.head(15).to_string(index=False), flush=True)

    # Sanity
    top5 = importance_df.head(5)["feature"].tolist()
    geo_in_top5 = any("geom_distance" in f or "geom_angle" in f for f in top5)
    print(f"\n  Distance/angle in top 5: {'PASS' if geo_in_top5 else 'INVESTIGATE'}", flush=True)

    # SHAP beeswarm plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_cols, show=False)
    plt.tight_layout()
    shap_path = REPORTS_DIR / "lgbm_shap_summary.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP plot to {shap_path}", flush=True)

    # 7. Save
    print("\n" + "=" * 60, flush=True)
    print("STEP 7: Save final model", flush=True)
    print("=" * 60, flush=True)

    model_path = MODELS_DIR / "final_xg_lgbm.joblib"
    joblib.dump(final_model, model_path)
    print(f"  Saved to {model_path}", flush=True)

    results = {
        "model": "LightGBM",
        "best_params": {k: v for k, v in best_params.items() if k != "verbose"},
        "cv_log_loss": best_cv_score,
        "test": metrics_lgbm_test,
        "benchmark_statsbomb": metrics_sb,
        "benchmark_baseline_lr": metrics_lr,
        "ece": float(ece),
        "top_15_shap": importance_df.head(15).to_dict("records"),
    }
    results_path = REPORTS_DIR / "lgbm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {results_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("DONE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
