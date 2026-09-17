"""
Data Ingestion and Validation Module
====================================

Handles CSV loading, column validation, data-type coercion, infinity/NaN auditing,
and class-balance ratio calculations.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from .config import WorkflowConfig


def load_and_validate_data(
    csv_path: Path,
    config: WorkflowConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads point observations from CSV, strips column whitespace, and validates fields.

    Parameters
    ----------
    csv_path : Path
        Path to the points CSV file.
    config : WorkflowConfig
        Configuration object specifying required columns.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]
        - Full loaded dataframe (df)
        - Raw predictor matrix (X_raw)
        - Target label vector (y)
        - Geographic coordinate table (coordinates)

    Raises
    ------
    FileNotFoundError
        If csv_path does not exist.
    ValueError
        If required fields are missing, target has unexpected labels, or coordinates contain NaNs.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found at: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    missing = [f for f in config.required_fields if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required fields in dataset ({csv_path.name}):\n"
            + "\n".join(f"  - {f}" for f in missing)
            + f"\nFound columns: {list(df.columns)}"
        )

    # Validate target labels {0, 1}
    y = pd.to_numeric(df[config.target_field], errors="raise").astype(int)
    unexpected_labels = set(y.unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(
            f"Target field '{config.target_field}' contains unexpected labels: {unexpected_labels}. "
            "Only binary values {0, 1} are supported."
        )

    # Build predictor matrix with proper numerical conversion
    X_raw = df[config.raw_features].copy()
    for field in config.numeric_features:
        X_raw[field] = pd.to_numeric(X_raw[field], errors="coerce")

    for field in config.categorical_features:
        X_raw[field] = pd.to_numeric(X_raw[field], errors="coerce")

    # Validate coordinates
    coordinates = df[config.coordinate_fields].apply(pd.to_numeric, errors="coerce")
    if coordinates.isna().any().any():
        raise ValueError(
            f"Coordinate fields {config.coordinate_fields} contain missing (NaN) values."
        )

    return df, X_raw, y, coordinates


def audit_predictor_quality(
    X_raw: pd.DataFrame,
    config: WorkflowConfig,
) -> pd.DataFrame:
    """
    Scans every predictor for missing (NaN), positive/negative infinity, and finite counts.

    Parameters
    ----------
    X_raw : pd.DataFrame
        Predictor matrix.
    config : WorkflowConfig
        Configuration object.

    Returns
    -------
    pd.DataFrame
        Table summarizing quality counts per feature.
    """
    rows = []
    for feature in config.raw_features:
        values = pd.to_numeric(X_raw[feature], errors="coerce").to_numpy(dtype=float)
        rows.append({
            "feature": feature,
            "missing_count": int(np.isnan(values).sum()),
            "positive_infinity_count": int(np.isposinf(values).sum()),
            "negative_infinity_count": int(np.isneginf(values).sum()),
            "finite_count": int(np.isfinite(values).sum()),
        })
    return pd.DataFrame(rows)


def compute_class_distribution(y: pd.Series) -> pd.DataFrame:
    """
    Computes class counts and mapped names for the inventory labels.
    """
    dist = (
        y.value_counts()
        .sort_index()
        .rename_axis("label")
        .reset_index(name="point_count")
    )
    dist["class_name"] = dist["label"].map({
        0: "Non-Landslide",
        1: "Landslide",
    })
    return dist


def compute_class_weights(y: pd.Series) -> float:
    """
    Calculates scale_pos_weight = negative_count / positive_count for XGBoost.
    """
    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())

    if positive_count == 0:
        raise ValueError("Cannot calculate scale_pos_weight: No positive (landslide) records found.")

    return float(negative_count / positive_count)
