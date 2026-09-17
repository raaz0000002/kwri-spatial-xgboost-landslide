"""
Workflow Configuration Module
=============================

Centralizes model hyperparameters, spatial partitioning settings, feature definitions,
and directory paths for reproducible and scalable execution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class WorkflowConfig:
    """Configuration parameters for the KWRI Spatial XGBoost Landslide Workflow."""

    # Random Seed
    seed: int = 42

    # Spatial Partitioning Parameters
    number_of_spatial_groups: int = 10
    number_of_cv_folds: int = 5
    spatial_holdout_group_fraction: float = 0.30

    # Decision Threshold
    classification_threshold: float = 0.50

    # Missing Value Handling
    missing_landcover_code: int = -9999

    # Model Hyperparameters (XGBoost)
    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 3
    min_child_weight: int = 2
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    n_jobs: int = 4

    # Target & Coordinate Columns
    target_field: str = "Labels"
    coordinate_fields: List[str] = field(
        default_factory=lambda: ["POINT_X", "POINT_Y"]
    )

    # Environmental & Topographic Predictors
    numeric_features: List[str] = field(
        default_factory=lambda: [
            "Curvature",
            "TWI",
            "Dist_Road_KYTC_1",
            "Plan_Curv",
            "Profile_Curv",
            "Distan_to_Rivers",
            "Slope_KWRI",
            "Aspect_KWRI",
            "Rain_Mean_KWRI",
            "NDVI_Mean_KWRI",
        ]
    )

    # Categorical Predictors
    categorical_features: List[str] = field(
        default_factory=lambda: ["Landcover_KWRI"]
    )

    # Input and Output Paths
    input_csv: Optional[Path] = None
    output_dir: Path = field(
        default_factory=lambda: Path("outputs") / "XGBoost_KWRI_Spatial_Final"
    )

    @property
    def raw_features(self) -> List[str]:
        """Combined list of numeric and categorical predictors."""
        return self.numeric_features + self.categorical_features

    @property
    def required_fields(self) -> List[str]:
        """All mandatory fields that must be present in the dataset."""
        return [self.target_field] + self.coordinate_fields + self.raw_features
