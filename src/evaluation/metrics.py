"""
Evaluation metrics and calibration visualization for xG models.

Priority order (per CLAUDE.md):
1. Log-loss (primary)
2. Brier score
3. Calibration curve / reliability diagram
4. ROC-AUC (secondary)
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all xG-relevant metrics.

    Args:
        y_true: Binary ground truth (0/1).
        y_prob: Predicted probabilities [0, 1].

    Returns:
        Dict with log_loss, brier_score, roc_auc.
    """
    return {
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def print_metrics_comparison(
    metrics_model: dict,
    metrics_benchmark: dict,
    model_name: str = "Our Model",
    benchmark_name: str = "StatsBomb xG",
) -> None:
    """Print side-by-side comparison table."""
    print(f"\n{'Metric':<15} {model_name:<15} {benchmark_name:<15} {'Winner':<15}")
    print("-" * 60)

    for key in ["log_loss", "brier_score", "roc_auc"]:
        m = metrics_model[key]
        b = metrics_benchmark[key]
        # Lower is better for log_loss and brier; higher for auc
        if key in ("log_loss", "brier_score"):
            winner = model_name if m < b else benchmark_name
        else:
            winner = model_name if m > b else benchmark_name
        print(f"{key:<15} {m:<15.6f} {b:<15.6f} {winner}")


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_model: np.ndarray,
    y_prob_benchmark: Optional[np.ndarray] = None,
    model_name: str = "Our Model",
    benchmark_name: str = "StatsBomb xG",
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot calibration (reliability) diagram.

    The closer to the diagonal, the better calibrated.

    Args:
        y_true: Binary ground truth.
        y_prob_model: Model predicted probabilities.
        y_prob_benchmark: Optional benchmark probabilities.
        model_name: Label for our model.
        benchmark_name: Label for benchmark.
        n_bins: Number of calibration bins.
        ax: Optional matplotlib axes.

    Returns:
        Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)

    # Our model
    prob_true, prob_pred = calibration_curve(y_true, y_prob_model, n_bins=n_bins)
    ax.plot(prob_pred, prob_true, "o-", color="#e74c3c", linewidth=2,
            markersize=6, label=model_name)

    # Benchmark
    if y_prob_benchmark is not None:
        prob_true_b, prob_pred_b = calibration_curve(
            y_true, y_prob_benchmark, n_bins=n_bins
        )
        ax.plot(prob_pred_b, prob_true_b, "s-", color="#3498db", linewidth=2,
                markersize=6, label=benchmark_name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title("Calibration / Reliability Diagram")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Our Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot predicted probability distributions split by actual outcome."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="#3498db",
            label="No Goal", density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color="#e74c3c",
            label="Goal", density=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name}: Predicted probability by outcome")
    ax.legend()
    ax.set_xlim([0, 1])

    return fig
