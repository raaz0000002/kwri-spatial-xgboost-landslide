"""
Kavrepalanchowk Landslide Susceptibility Package
=================================================

Spatially validated machine learning framework for regional landslide susceptibility
modeling in Kavrepalanchowk District, Nepal Middle Hills.

Features:
- Spatially blocked cross-validation (10-km blocks) to avoid spatial autocorrelation bias
- Yeo-Johnson power transforms fitted fold-by-fold
- Leakage-safe predictor screening (excluding post-event spectral indices)
- Ensemble modeling (RF, XGBoost, GBM, SVM) with Platt probability calibration
- Fast DeLong ROC-AUC covariance tests with Holm-Bonferroni correction
- Area of Applicability (AOA) multivariate dissimilarity analysis
"""

from .common import (
    RAW,
    SEED,
    CONT,
    FAMILY,
    SOIL_LEVELS,
    build_X,
    family_of,
    y,
    XY,
    SkewYJ,
    square_blocks,
)
from .delong import (
    delong_cov,
    delong_pair,
    auc_ci,
    holm,
)
from .figstyle import (
    CAT,
    save,
    map_furniture,
)
from .docxwriter import write as write_docx

__version__ = "2.0.0"

__all__ = [
    "RAW",
    "SEED",
    "CONT",
    "FAMILY",
    "SOIL_LEVELS",
    "build_X",
    "family_of",
    "y",
    "XY",
    "SkewYJ",
    "square_blocks",
    "delong_cov",
    "delong_pair",
    "auc_ci",
    "holm",
    "CAT",
    "save",
    "map_furniture",
    "write_docx",
]
