"""
Automated Unit and Integration Tests for KWRI Landslide Workflow
===============================================================
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path for direct imports during testing
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from kwri_landslide.config import WorkflowConfig
from kwri_landslide.data import (
    load_and_validate_data,
    audit_predictor_quality,
    compute_class_weights,
    compute_class_distribution,
)
from kwri_landslide.spatial import create_spatial_blocks, split_spatial_holdout
from kwri_landslide.pipeline import build_spatial_pipeline
from kwri_landslide.explainability import (
    readable_feature_name,
    safe_reporting_name,
    compute_feature_importance,
    aggregate_landcover_importance,
)


class TestKWRIWorkflow(unittest.TestCase):
    """Test suite verifying components of the spatial ML pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.config = WorkflowConfig()
        cls.sample_csv = ROOT_DIR / "data" / "sample_test_points.csv"
        cls.assertTrue(cls.sample_csv.exists(), f"Sample CSV not found at {cls.sample_csv}")

    def test_config_initialization(self):
        """Test default configurations and derived properties."""
        cfg = WorkflowConfig()
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.number_of_spatial_groups, 10)
        self.assertEqual(cfg.number_of_cv_folds, 5)
        self.assertEqual(cfg.classification_threshold, 0.50)
        self.assertIn("Curvature", cfg.numeric_features)
        self.assertIn("Landcover_KWRI", cfg.categorical_features)
        self.assertEqual(len(cfg.raw_features), 11)
        self.assertEqual(len(cfg.required_fields), 14)

    def test_data_loading_and_validation(self):
        """Test loading and validating sample points dataset."""
        df, X_raw, y, coords = load_and_validate_data(self.sample_csv, self.config)
        self.assertGreater(len(df), 50)
        self.assertEqual(len(X_raw), len(df))
        self.assertEqual(len(y), len(df))
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(coords.shape[1], 2)
        self.assertFalse(coords.isna().any().any())

    def test_missing_fields_validation_error(self):
        """Test that missing required columns triggers a ValueError."""
        df = pd.read_csv(self.sample_csv).drop(columns=["Curvature"])
        temp_csv = ROOT_DIR / "data" / "_temp_invalid.csv"
        df.to_csv(temp_csv, index=False)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_data(temp_csv, self.config)
            self.assertIn("Missing required fields", str(ctx.exception))
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    def test_audit_predictor_quality(self):
        """Test data quality audit table."""
        _, X_raw, _, _ = load_and_validate_data(self.sample_csv, self.config)
        quality_df = audit_predictor_quality(X_raw, self.config)
        self.assertEqual(len(quality_df), len(self.config.raw_features))
        self.assertIn("missing_count", quality_df.columns)
        self.assertIn("finite_count", quality_df.columns)

    def test_class_weights_and_distribution(self):
        """Test class distribution calculation and scale_pos_weight."""
        _, _, y, _ = load_and_validate_data(self.sample_csv, self.config)
        dist = compute_class_distribution(y)
        self.assertEqual(len(dist), 2)
        weight = compute_class_weights(y)
        self.assertGreater(weight, 0.0)

    def test_spatial_clustering_and_split(self):
        """Test K-Means spatial grouping and GroupShuffleSplit."""
        _, X_raw, y, coords = load_and_validate_data(self.sample_csv, self.config)
        groups = create_spatial_blocks(coords, n_clusters=5, seed=42)
        self.assertEqual(len(groups), len(coords))
        self.assertEqual(len(np.unique(groups)), 5)

        train_idx, test_idx = split_spatial_holdout(X_raw, y, groups, test_size=0.40, seed=42)
        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(test_idx), 0)
        # Verify complete cluster isolation between train and test
        train_clusters = set(groups[train_idx])
        test_clusters = set(groups[test_idx])
        self.assertEqual(train_clusters.intersection(test_clusters), set())

    def test_pipeline_construction_and_mini_fit(self):
        """Test building pipeline, fitting on mini dataset, and predicting probabilities."""
        _, X_raw, y, _ = load_and_validate_data(self.sample_csv, self.config)
        weight = compute_class_weights(y)
        pipeline = build_spatial_pipeline(self.config, scale_pos_weight=weight)

        # Fit model
        pipeline.fit(X_raw, y)
        probs = pipeline.predict_proba(X_raw)[:, 1]
        self.assertEqual(len(probs), len(X_raw))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        # Feature importance check
        importance_df = compute_feature_importance(pipeline, self.config)
        self.assertGreater(len(importance_df), 0)
        self.assertIn("importance_percent", importance_df.columns)

        grouped_df = aggregate_landcover_importance(importance_df)
        self.assertIn("Land Cover Total", grouped_df["grouped_feature"].values)


if __name__ == "__main__":
    unittest.main()
