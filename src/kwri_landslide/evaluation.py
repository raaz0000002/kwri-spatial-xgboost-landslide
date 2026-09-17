"""
Model Evaluation and Validation Module
======================================

Implements spatial cross-validation (GroupKFold), pooled Out-Of-Fold (OOF) prediction,
independent spatial holdout assessment, detailed classification reporting,
annotated confusion matrix extraction, and ROC curve generation.
"""

from typing import Callable, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline


def run_spatial_cross_validation(
    pipeline_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    spatial_groups: np.ndarray,
    n_splits: int = 5,
    threshold: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Executes K-fold spatial cross-validation using GroupKFold and computes pooled out-of-fold predictions.

    Parameters
    ----------
    pipeline_factory : Callable[[], Pipeline]
        Factory function returning a fresh unfitted Pipeline.
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Target vector.
    spatial_groups : np.ndarray
        Spatial group assignments.
    n_splits : int
        Number of spatial folds. Default is 5.
    threshold : float
        Decision threshold for class prediction. Default is 0.50.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]
        - cv_summary (table with fold scores, mean, std)
        - oof_metrics (pooled out-of-fold performance summary)
        - oof_probability (pooled out-of-fold susceptibility scores)
        - oof_prediction (pooled predicted binary classes)
    """
    spatial_cv = GroupKFold(n_splits=n_splits)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    cv_result = cross_validate(
        pipeline_factory(),
        X,
        y,
        groups=spatial_groups,
        cv=spatial_cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    # Collect per-fold details
    fold_details = []
    for fold_number, (train_idx, test_idx) in enumerate(
        spatial_cv.split(X, y, groups=spatial_groups), start=1
    ):
        test_groups = sorted(np.unique(spatial_groups[test_idx]).astype(int).tolist())
        test_labels = y.iloc[test_idx]
        fold_details.append({
            "fold": fold_number,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "test_groups": ",".join(map(str, test_groups)),
            "test_non_landslide": int((test_labels == 0).sum()),
            "test_landslide": int((test_labels == 1).sum()),
        })
    fold_details_df = pd.DataFrame(fold_details)

    cv_table = pd.DataFrame({
        "fold": range(1, n_splits + 1),
        "accuracy": cv_result["test_accuracy"],
        "precision": cv_result["test_precision"],
        "recall": cv_result["test_recall"],
        "f1": cv_result["test_f1"],
        "roc_auc": cv_result["test_roc_auc"],
    }).merge(fold_details_df, on="fold", how="left")

    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_mean = cv_table[metric_columns].mean()
    cv_std = cv_table[metric_columns].std(ddof=1)

    cv_summary = pd.concat([
        cv_table,
        pd.DataFrame([
            {"fold": "mean", **cv_mean.to_dict()},
            {"fold": "std", **cv_std.to_dict()},
        ]),
    ], ignore_index=True)

    # Pooled out-of-fold predictions
    oof_probability = cross_val_predict(
        pipeline_factory(),
        X,
        y,
        groups=spatial_groups,
        cv=spatial_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    oof_prediction = (oof_probability >= threshold).astype(int)

    oof_metrics = pd.DataFrame([{
        "accuracy": accuracy_score(y, oof_prediction),
        "precision": precision_score(y, oof_prediction, zero_division=0),
        "recall": recall_score(y, oof_prediction, zero_division=0),
        "f1": f1_score(y, oof_prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, oof_probability),
        "prediction_type": "SPATIAL_OUT_OF_FOLD_VALIDATION",
    }])

    return cv_summary, oof_metrics, oof_probability, oof_prediction


def evaluate_holdout(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_groups: list,
    test_groups: list,
    threshold: float = 0.50,
) -> Tuple[Pipeline, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Fits pipeline on spatial training partition and evaluates on withheld spatial holdout partition.

    Returns
    -------
    Tuple[Pipeline, pd.DataFrame, np.ndarray, np.ndarray]
        - Fitted model
        - Metrics summary dataframe
        - Probability scores
        - Binary class predictions
    """
    model.fit(X_train, y_train)

    holdout_probability = model.predict_proba(X_test)[:, 1]
    holdout_prediction = (holdout_probability >= threshold).astype(int)

    acc = accuracy_score(y_test, holdout_prediction)
    prec = precision_score(y_test, holdout_prediction, zero_division=0)
    rec = recall_score(y_test, holdout_prediction, zero_division=0)
    f1 = f1_score(y_test, holdout_prediction, zero_division=0)
    auc = roc_auc_score(y_test, holdout_probability)

    metrics_df = pd.DataFrame([{
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "classification_threshold": threshold,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "training_groups": ",".join(map(str, train_groups)),
        "testing_groups": ",".join(map(str, test_groups)),
        "test_non_landslide": int((y_test == 0).sum()),
        "test_landslide": int((y_test == 1).sum()),
        "prediction_type": "SPATIAL_GROUP_HOLDOUT",
    }])

    return model, metrics_df, holdout_probability, holdout_prediction


def compute_explained_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Computes confusion matrix, contingency counts (TP, TN, FP, FN), specificity, and classification report.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    cm_df = pd.DataFrame(
        cm,
        index=["Observed_Non-Landslide", "Observed_Landslide"],
        columns=["Predicted_Non-Landslide", "Predicted_Landslide"],
    )

    explained_metrics = pd.DataFrame([{
        "true_negative_TN": int(tn),
        "false_positive_FP": int(fp),
        "false_negative_FN": int(fn),
        "true_positive_TP": int(tp),
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": auc,
    }])

    metrics_dict = {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "accuracy": acc,
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Non-Landslide", "Landslide"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    return cm, cm_df, explained_metrics, metrics_dict, report_df


def compute_roc_curve_data(
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Calculates False Positive Rate, True Positive Rate, and ROC threshold table.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_df = pd.DataFrame({
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "threshold": thresholds,
    })
    return fpr, tpr, roc_df
