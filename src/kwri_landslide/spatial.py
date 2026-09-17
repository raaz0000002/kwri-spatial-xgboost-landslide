"""
Spatial Partitioning Module
===========================

Implements spatial blocking via K-Means coordinate clustering and GroupShuffleSplit
to guard against spatial autocorrelation leakage during validation.
"""

import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit


def create_spatial_blocks(
    coordinates: pd.DataFrame,
    n_clusters: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Partitions spatial coordinates into contiguous geographic clusters using K-Means.

    Parameters
    ----------
    coordinates : pd.DataFrame
        Table with POINT_X and POINT_Y coordinates.
    n_clusters : int
        Number of geographic clusters to create. Default is 10.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        Array of integer spatial cluster IDs for each point.
    """
    warnings.filterwarnings(
        "ignore",
        message="KMeans is known to have a memory leak on Windows.*",
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=seed,
    )
    return kmeans.fit_predict(coordinates.to_numpy())


def split_spatial_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    spatial_groups: np.ndarray,
    test_size: float = 0.30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates an independent spatial holdout split using GroupShuffleSplit.
    Ensures that entire spatial clusters are either in training or testing, never both.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Target vector.
    spatial_groups : np.ndarray
        Spatial cluster assignments for each point.
    test_size : float
        Fraction of spatial groups to allocate to holdout. Default is 0.30.
    seed : int
        Random seed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=spatial_groups))

    # Sanity checks
    if y.iloc[train_idx].nunique() < 2:
        raise ValueError("Spatial holdout training partition contains only one class.")
    if y.iloc[test_idx].nunique() < 2:
        raise ValueError("Spatial holdout testing partition contains only one class.")

    return train_idx, test_idx
