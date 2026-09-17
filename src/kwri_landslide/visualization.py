"""
Visualization Module
====================

Generates publication-quality (300 DPI) scientific visualizations with clean,
consistent aesthetics, color palettes, and typography.
"""

from pathlib import Path
from typing import Dict, List, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for scripts and servers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_class_distribution(
    class_distribution: pd.DataFrame,
    output_path: Path,
) -> None:
    """Renders and saves the landslide vs. non-landslide distribution bar chart."""
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        class_distribution["class_name"],
        class_distribution["point_count"],
        color=["#4C78A8", "#C44E52"],
    )
    ax.set_ylabel("Number of Points")
    ax.set_title("Landslide Inventory Class Distribution", fontweight="bold")
    ax.bar_label(bars, padding=4)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cv_metrics(
    cv_mean: pd.Series,
    cv_std: pd.Series,
    metric_columns: List[str],
    output_path: Path,
) -> None:
    """Renders and saves the spatial cross-validation error-bar performance plot."""
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    bars = ax.bar(
        labels,
        cv_mean[metric_columns],
        yerr=cv_std[metric_columns],
        capsize=5,
        color="#326273",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spatial Cross-Validation Score")
    ax.set_title("Five-Fold Spatial Cross-Validation Performance", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in cv_mean[metric_columns]], padding=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_explained_confusion_matrix(
    confusion_matrix_data: Any,
    metrics_dict: Dict[str, Any],
    output_path: Path,
) -> None:
    """Renders and saves the annotated confusion matrix heatmap with contingency callout box."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.5,
        linecolor="#D0D0D0",
        annot_kws={"fontsize": 16},
        xticklabels=["Non-Landslide", "Landslide"],
        yticklabels=["Non-Landslide", "Landslide"],
        ax=ax,
    )
    ax.set_title(
        f"XGBoost Spatial Holdout\nAUC={metrics_dict['auc']:.4f} | Acc={metrics_dict['accuracy']:.4f}",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    metrics_text = (
        f"TP = {metrics_dict['tp']}\n"
        f"TN = {metrics_dict['tn']}\n"
        f"FP = {metrics_dict['fp']}\n"
        f"FN = {metrics_dict['fn']}\n\n"
        f"Precision = {metrics_dict['precision']:.3f}\n"
        f"Recall = {metrics_dict['recall']:.3f}\n"
        f"Specificity = {metrics_dict['specificity']:.3f}\n"
        f"F1-Score = {metrics_dict['f1']:.3f}"
    )
    ax.text(
        1.08,
        0.50,
        metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="center",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#FFFDE7",
            "edgecolor": "#333333",
            "linewidth": 1.0,
        },
    )
    fig.subplots_adjust(left=0.13, right=0.73, bottom=0.13, top=0.82)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_roc_curve(
    fpr: Any,
    tpr: Any,
    auc_score: float,
    output_path: Path,
) -> None:
    """Renders and saves the spatial holdout ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#C44E52", lw=2, label=f"Spatial holdout AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], color="#4C78A8", linestyle="--", label="Random Chance (0.50)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("XGBoost Spatial Holdout ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_feature_importance(
    df: pd.DataFrame,
    feature_col: str,
    importance_col: str,
    colors: List[str],
    common_axis_max: float,
    title: str,
    output_path: Path,
    fig_height: float = 7.0,
) -> None:
    """Renders and saves horizontal bar charts of predictor importance on a shared scale."""
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(df[feature_col], df[importance_col], color=colors)
    ax.set_xlim(0, common_axis_max)
    ax.set_xlabel("XGBoost Feature Importance")
    ax.set_ylabel("")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.30)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in df[importance_col]], padding=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
