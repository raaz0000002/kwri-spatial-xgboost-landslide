"""
Model Explainability and Feature Importance Module
==================================================

Translates one-hot encoded and raw XGBoost booster split contributions into
human-readable names, aggregates NLCD Land Cover classes into a unified total,
and ranks predictor contributions.
"""

import re
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import WorkflowConfig

NLCD_LANDCOVER_NAMES = {
    -9999: "Missing/Unknown",
    0: "NoData / Unclassified",
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    51: "Dwarf Scrub",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    72: "Sedge/Herbaceous",
    73: "Lichens",
    74: "Moss",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}

REGULAR_FEATURE_NAMES = {
    "Curvature": "Curvature",
    "TWI": "Topographic Wetness Index (TWI)",
    "Dist_Road_KYTC_1": "Distance to KYTC Roads",
    "Plan_Curv": "Plan Curvature",
    "Profile_Curv": "Profile Curvature",
    "Distan_to_Rivers": "Distance to Rivers/Streams",
    "Slope_KWRI": "Slope",
    "Aspect_KWRI": "Aspect",
    "Rain_Mean_KWRI": "Mean Rainfall",
    "NDVI_Mean_KWRI": "NDVI",
}


def readable_feature_name(feature: str) -> str:
    """Translates encoded booster feature names into readable labels."""
    feature = str(feature)
    prefix = "Landcover_KWRI_"
    if feature.startswith(prefix):
        raw_code = feature[len(prefix):]
        try:
            code = int(float(raw_code))
            class_name = NLCD_LANDCOVER_NAMES.get(code, "Unknown NLCD Class")
            return f"Land Cover: {code} ({class_name})"
        except (ValueError, TypeError):
            return "Land Cover: Missing/Unknown"
    return REGULAR_FEATURE_NAMES.get(feature, feature)


def safe_reporting_name(name: str) -> str:
    """Produces a file-safe reporting name string."""
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(name))
    return re.sub(r"_+", "_", cleaned).strip("_") or "feature"


def compute_feature_importance(
    pipeline: Pipeline,
    config: WorkflowConfig,
) -> pd.DataFrame:
    """
    Extracts raw feature importances from the fitted XGBoost booster in the pipeline.

    Returns
    -------
    pd.DataFrame
        Detailed feature importance ranking table.
    """
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_classifier = pipeline.named_steps["classifier"]
    encoded_feature_names = fitted_preprocessor.get_feature_names_out()

    df = pd.DataFrame({
        "feature": encoded_feature_names,
        "importance": fitted_classifier.feature_importances_,
    })
    df["safe_feature_name"] = df["feature"].map(safe_reporting_name)
    df["is_landcover"] = df["feature"].str.startswith("Landcover_KWRI_")
    df["display_feature"] = df["feature"].map(readable_feature_name)
    df["importance_percent"] = df["importance"] * 100
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def aggregate_landcover_importance(
    detailed_importance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combines all one-hot encoded NLCD Land Cover classes into a single 'Land Cover Total' metric.

    Returns
    -------
    pd.DataFrame
        Grouped feature importance ranking table.
    """
    df = detailed_importance.copy()
    df["grouped_feature"] = np.where(
        df["is_landcover"], "Land Cover Total", df["display_feature"]
    )
    grouped = (
        df.groupby("grouped_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["importance_percent"] = grouped["importance"] * 100
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped
