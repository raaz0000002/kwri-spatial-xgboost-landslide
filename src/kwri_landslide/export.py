"""
Artifact Export and Metadata Serialization Module
=================================================

Handles saving trained scikit-learn/XGBoost pipelines, comprehensive model bundles,
JSON run settings, and results inventory tables.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import WorkflowConfig


def save_model_bundle(
    pipeline: Pipeline,
    config: WorkflowConfig,
    scale_pos_weight: float,
    output_path: Path,
) -> None:
    """
    Serializes a self-contained model bundle dictionary to disk with joblib.
    """
    bundle = {
        "pipeline": pipeline,
        "raw_features": config.raw_features,
        "numeric_features": config.numeric_features,
        "categorical_features": config.categorical_features,
        "missing_landcover_code": config.missing_landcover_code,
        "classification_threshold": config.classification_threshold,
        "scale_pos_weight": scale_pos_weight,
        "number_of_spatial_groups": config.number_of_spatial_groups,
        "number_of_cv_folds": config.number_of_cv_folds,
        "random_seed": config.seed,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def export_run_metadata(
    config: WorkflowConfig,
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Writes structured experiment metadata and settings to a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input_csv": str(config.input_csv.resolve()) if config.input_csv else None,
        "output_folder": str(config.output_dir.resolve()),
        "number_of_records": stats.get("n_records"),
        "class_counts": stats.get("class_counts", {}),
        "scale_pos_weight": stats.get("scale_pos_weight"),
        "raw_features": config.raw_features,
        "number_of_spatial_groups": config.number_of_spatial_groups,
        "number_of_cv_folds": config.number_of_cv_folds,
        "spatial_holdout_group_fraction": config.spatial_holdout_group_fraction,
        "spatial_holdout_training_groups": stats.get("holdout_train_groups", []),
        "spatial_holdout_testing_groups": stats.get("holdout_test_groups", []),
        "classification_threshold": config.classification_threshold,
        "random_seed": config.seed,
        "primary_evaluation": "five_fold_spatial_cross_validation",
        "secondary_evaluation": "spatial_group_holdout",
        "mapping_pipeline": stats.get("mapping_pipeline_path"),
        "model_bundle": stats.get("model_bundle_path"),
        "all_point_prediction_warning": "IN_SAMPLE_NOT_VALIDATION",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def build_results_inventory(output_dir: Path) -> pd.DataFrame:
    """
    Scans the results directory and produces an audit DataFrame of all generated artifacts.
    """
    output_dir = Path(output_dir)
    files = sorted(
        [p for p in output_dir.iterdir() if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    return pd.DataFrame({
        "file_name": [p.name for p in files],
        "file_size_bytes": [p.stat().st_size for p in files],
        "full_path": [str(p.resolve()) for p in files],
    })
