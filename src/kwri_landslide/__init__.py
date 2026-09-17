"""
KWRI Landslide Susceptibility Mapping Package
============================================

Spatial machine learning workflow for regional landslide susceptibility modeling
using XGBoost, spatial K-Means blocking, and leakage-safe pipelines.
"""

from .config import WorkflowConfig
from .data import load_and_validate_data, audit_predictor_quality, compute_class_weights
from .spatial import create_spatial_blocks, split_spatial_holdout
from .pipeline import build_spatial_pipeline
from .evaluation import (
    run_spatial_cross_validation,
    evaluate_holdout,
    compute_explained_confusion_matrix,
)
from .explainability import (
    compute_feature_importance,
    aggregate_landcover_importance,
    NLCD_LANDCOVER_NAMES,
)
from .visualization import (
    plot_class_distribution,
    plot_cv_metrics,
    plot_explained_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)
from .export import save_model_bundle, export_run_metadata, build_results_inventory

__version__ = "1.0.0"

__all__ = [
    "WorkflowConfig",
    "load_and_validate_data",
    "audit_predictor_quality",
    "compute_class_weights",
    "create_spatial_blocks",
    "split_spatial_holdout",
    "build_spatial_pipeline",
    "run_spatial_cross_validation",
    "evaluate_holdout",
    "compute_explained_confusion_matrix",
    "compute_feature_importance",
    "aggregate_landcover_importance",
    "NLCD_LANDCOVER_NAMES",
    "plot_class_distribution",
    "plot_cv_metrics",
    "plot_explained_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
    "save_model_bundle",
    "export_run_metadata",
    "build_results_inventory",
]
