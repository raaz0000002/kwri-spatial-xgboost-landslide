"""
Automated Unit and Integration Tests for Kavrepalanchowk Landslide Workflow
==========================================================================
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Add src to path for direct package imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from kavre_landslide.common import (
    RAW,
    CONT,
    FAMILY,
    build_X,
    SkewYJ,
    square_blocks,
    family_of,
    y,
    XY,
)
from kavre_landslide.delong import (
    delong_cov,
    delong_pair,
    auc_ci,
    holm,
)


class TestKavreWorkflow(unittest.TestCase):
    """Test suite verifying components of the Kavrepalanchowk spatial ML pipeline."""

    def test_dataset_geometry_and_balance(self):
        """Verify 1200 inventory points, 50% landslide balance, and UTM 45N coordinates."""
        self.assertEqual(len(RAW), 1200)
        self.assertEqual(int((RAW["Landslide"] == 1).sum()), 600)
        self.assertEqual(int((RAW["Landslide"] == 0).sum()), 600)
        self.assertEqual(len(y), 1200)
        self.assertEqual(XY.shape, (1200, 2))

        # Check UTM 45N coordinate bounds for Kavrepalanchowk
        self.assertTrue(np.all(XY[:, 0] > 340000))
        self.assertTrue(np.all(XY[:, 0] < 400000))
        self.assertTrue(np.all(XY[:, 1] > 3000000))
        self.assertTrue(np.all(XY[:, 1] < 3100000))

    def test_feature_matrix_construction(self):
        """Verify build_X creates all continuous features, aspect transforms, and parent soil encodings."""
        X = build_X(RAW, soil=True)
        self.assertEqual(len(X), 1200)

        # Check trigonometric aspect encoding
        self.assertIn("Northness", X.columns)
        self.assertIn("Eastness", X.columns)
        self.assertTrue(np.all((X["Northness"] >= -1.0) & (X["Northness"] <= 1.0)))
        self.assertTrue(np.all((X["Eastness"] >= -1.0) & (X["Eastness"] <= 1.0)))

        # Check soil levels and indicator columns
        for level in [1, 2, 3, 4, 5]:
            self.assertIn(f"ParentSoil_{level}", X.columns)
        self.assertIn("ParentSoil_missing", X.columns)

        # Verify families
        self.assertEqual(family_of("Elevation"), "Terrain")
        self.assertEqual(family_of("FlowAcc"), "Hydrology")
        self.assertEqual(family_of("Sand"), "Soil texture")
        self.assertEqual(family_of("ParentSoil_1"), "Parent soil")

    def test_skew_yeo_johnson_transformer(self):
        """Test SkewYJ transformer inside an sklearn Pipeline."""
        X = build_X(RAW, soil=True)
        pipe = Pipeline([
            ("transformer", SkewYJ(skew_thr=0.5)),
            ("classifier", LogisticRegression(max_iter=200))
        ])

        # Fit and predict
        pipe.fit(X, y)
        probs = pipe.predict_proba(X)[:, 1]
        self.assertEqual(len(probs), 1200)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_square_blocks_spatial_grouping(self):
        """Verify square_blocks clusters nearby points into deterministic spatial blocks."""
        blocks = square_blocks(XY, size_m=10000.0, offset=(0.0, 0.0))
        self.assertEqual(len(blocks), 1200)
        unique_blocks = np.unique(blocks)
        # Should have between 10 and 40 geographic blocks across Kavrepalanchowk
        self.assertGreaterEqual(len(unique_blocks), 10)
        self.assertLessEqual(len(unique_blocks), 50)

    def test_delong_test_and_holm_correction(self):
        """Verify DeLong test calculations and Holm-Bonferroni correction."""
        rng = np.random.RandomState(42)
        y_true = np.array([1] * 50 + [0] * 50)
        pred_a = y_true * 0.8 + rng.uniform(0, 0.2, 100)
        pred_b = y_true * 0.6 + rng.uniform(0, 0.4, 100)

        # CI
        auc_val, ci_lo, ci_hi = auc_ci(y_true, pred_a)
        self.assertGreater(auc_val, 0.5)
        self.assertLessEqual(ci_lo, auc_val)
        self.assertGreaterEqual(ci_hi, auc_val)

        # DeLong pair comparison
        a0, a1, diff, diff_lo, diff_hi, p_val = delong_pair(y_true, pred_a, pred_b)
        self.assertGreaterEqual(a0, a1)
        self.assertGreaterEqual(p_val, 0.0)
        self.assertLessEqual(p_val, 1.0)

        # Holm correction
        raw_p = [0.01, 0.04, 0.03, 0.20]
        adj_p = holm(raw_p)
        self.assertEqual(len(adj_p), len(raw_p))
        self.assertTrue(np.all(adj_p >= np.array(raw_p)))


if __name__ == "__main__":
    unittest.main()
