"""
Pipeline Factory Module
=======================

Builds leakage-safe scikit-learn Pipelines combining preprocessing, imputation,
one-hot encoding, and XGBoost gradient boosting classifiers.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

from .config import WorkflowConfig


def build_spatial_pipeline(
    config: WorkflowConfig,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """
    Constructs a complete preprocessing and XGBoost classification pipeline.

    Preprocessing guarantees:
    - Numeric predictors: ±inf converted to NaN -> median imputation on training fold only.
    - Categorical predictors: ±inf converted to NaN -> missing code (-9999) -> one-hot encoding.
    - Leakage-safe: ColumnTransformer statistics are fitted strictly on the active training fold.

    Parameters
    ----------
    config : WorkflowConfig
        Configuration dataclass.
    scale_pos_weight : float
        Inverse class ratio (neg / pos) to penalize false negatives on imbalanced inventories.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline with steps [('preprocessor', ...), ('classifier', ...)].
    """
    numeric_pipeline = Pipeline([
        (
            "nonfinite_to_nan",
            FunctionTransformer(
                np.nan_to_num,
                kw_args={
                    "copy": True,
                    "nan": np.nan,
                    "posinf": np.nan,
                    "neginf": np.nan,
                },
                feature_names_out="one-to-one",
            ),
        ),
        (
            "median_imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    landcover_pipeline = Pipeline([
        (
            "nonfinite_to_nan",
            FunctionTransformer(
                np.nan_to_num,
                kw_args={
                    "copy": True,
                    "nan": np.nan,
                    "posinf": np.nan,
                    "neginf": np.nan,
                },
                feature_names_out="one-to-one",
            ),
        ),
        (
            "missing_category",
            SimpleImputer(
                strategy="constant",
                fill_value=config.missing_landcover_code,
            ),
        ),
        (
            "one_hot_encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.numeric_features),
            ("landcover", landcover_pipeline, config.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=config.seed,
        n_jobs=config.n_jobs,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
