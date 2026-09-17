"""
KWRI Spatial XGBoost Landslide Susceptibility Workflow Runner
============================================================

Unified CLI and Interactive Entrypoint for the KWRI Landslide Modeling Pipeline.
Supports both interactive (GUI file picker / prompt) and automated headless CLI execution.

Usage
-----
Interactive mode (GUI / Prompt):
    python run_workflow.py

Automated CLI execution (Headless / Scalable):
    python run_workflow.py --input data/sample_test_points.csv --output outputs/final_run --headless

CLI Options:
    --input, -i         Path to point observations CSV.
    --output, -o        Output directory for artifacts and figures.
    --clusters          Number of K-Means spatial clusters (default: 10).
    --folds             Number of spatial CV folds (default: 5).
    --holdout-frac      Fraction of spatial groups for holdout (default: 0.30).
    --threshold         Binary classification decision threshold (default: 0.50).
    --seed              Random state seed for reproducibility (default: 42).
    --headless          Disable interactive GUI and confirmation prompts.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path for direct modular execution
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# Set OpenMP threads to prevent CPU oversubscription
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from kwri_landslide.config import WorkflowConfig
from kwri_landslide.data import (
    load_and_validate_data,
    audit_predictor_quality,
    compute_class_distribution,
    compute_class_weights,
)
from kwri_landslide.spatial import create_spatial_blocks, split_spatial_holdout
from kwri_landslide.pipeline import build_spatial_pipeline
from kwri_landslide.evaluation import (
    run_spatial_cross_validation,
    evaluate_holdout,
    compute_explained_confusion_matrix,
    compute_roc_curve_data,
)
from kwri_landslide.explainability import (
    compute_feature_importance,
    aggregate_landcover_importance,
)
from kwri_landslide.visualization import (
    plot_class_distribution,
    plot_cv_metrics,
    plot_explained_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)
from kwri_landslide.export import (
    save_model_bundle,
    export_run_metadata,
    build_results_inventory,
)


def prompt_user_for_input(default_names=None) -> Path:
    """Interactively prompts the user to select or drag-and-drop the points CSV."""
    if default_names is None:
        default_names = [
            "KWRI_latest_points_export.csv",
            "data/sample_test_points.csv",
            "sample_test_points.csv",
            "kwri_points.csv",
        ]

    for candidate in default_names:
        cand_path = ROOT_DIR / candidate
        if cand_path.exists():
            print(f"\n[i] Detected existing dataset: {cand_path}")
            resp = input("Use this file? [Y/n]: ").strip().lower()
            if resp in ("", "y", "yes"):
                return cand_path

    # Try native graphical file dialog
    selected_path = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        print("\nOpening file selection window... Please choose your points CSV file.")
        file_chosen = filedialog.askopenfilename(
            title="Select Landslide Points CSV (e.g., KWRI_latest_points_export.csv)",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialdir=str(ROOT_DIR),
        )
        root.destroy()
        if file_chosen:
            selected_path = Path(file_chosen)
    except Exception:
        selected_path = None

    # Terminal fallback
    while not selected_path or not selected_path.exists():
        if selected_path and not selected_path.exists():
            print(f"\n[!] File not found: {selected_path}")
        user_input = input(
            "\nEnter path to CSV file (or drag & drop file here, or 'q' to quit): "
        ).strip().strip("\"'").strip()

        if user_input.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)

        if user_input:
            selected_path = Path(user_input)

    return selected_path


def run_pipeline(config: WorkflowConfig, headless: bool = False) -> None:
    """Executes the full 28-step KWRI Spatial ML Pipeline."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("KWRI SPATIAL XGBOOST LANDSLIDE SUSCEPTIBILITY WORKFLOW")
    print("=" * 75)
    print(f"Input Dataset:     {config.input_csv.resolve()}")
    print(f"Output Directory:  {output_dir.resolve()}")
    print(f"Random Seed:       {config.seed}")
    print(f"Spatial Blocking:  {config.number_of_spatial_groups} Clusters | {config.number_of_cv_folds}-Fold Spatial CV")
    print("-" * 75)

    # 1. Load and validate data
    print("[1/28] Loading and validating CSV points...")
    df, X_raw, y, coordinates = load_and_validate_data(config.input_csv, config)
    print(f"       Loaded {len(df)} records with {len(config.raw_features)} predictor features.")

    # 2. Predictor data quality report
    print("[2/28] Auditing predictor quality (NaN/inf inspection)...")
    quality_report = audit_predictor_quality(X_raw, config)
    quality_report.to_csv(output_dir / "xgboost_predictor_data_quality.csv", index=False)

    # 3. Class distribution & visualization
    print("[3/28] Analyzing class balance & generating distribution chart...")
    class_dist = compute_class_distribution(y)
    class_dist.to_csv(output_dir / "xgboost_class_distribution.csv", index=False)
    plot_class_distribution(class_dist, output_dir / "xgboost_class_distribution.png")

    # 4. Class-balance weight
    print("[4/28] Calculating scale_pos_weight for imbalanced formulation...")
    scale_pos_weight = compute_class_weights(y)
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    print(f"       Non-Landslide: {neg_count} | Landslide: {pos_count} | scale_pos_weight: {scale_pos_weight:.4f}")

    # 5. Spatial K-Means Blocking
    print(f"[5/28] Creating {config.number_of_spatial_groups} spatial blocks (K-Means on coordinates)...")
    spatial_groups = create_spatial_blocks(
        coordinates,
        n_clusters=config.number_of_spatial_groups,
        seed=config.seed,
    )
    spatial_assignments = df[["POINT_X", "POINT_Y", config.target_field]].copy()
    spatial_assignments["spatial_group"] = spatial_groups
    spatial_assignments.to_csv(output_dir / "xgboost_spatial_group_assignments.csv", index=False)

    crosstab = pd.crosstab(
        pd.Series(spatial_groups, name="spatial_group"),
        pd.Series(y.to_numpy(), name="label"),
        margins=True,
    )
    crosstab.to_csv(output_dir / "xgboost_spatial_group_balance.csv")

    # 6. Five-Fold Spatial Cross-Validation
    print(f"[6/28] Executing {config.number_of_cv_folds}-Fold Spatial Cross-Validation (GroupKFold)...")
    pipeline_factory = lambda: build_spatial_pipeline(config, scale_pos_weight=scale_pos_weight)
    cv_summary, oof_metrics, oof_probs, oof_preds = run_spatial_cross_validation(
        pipeline_factory=pipeline_factory,
        X=X_raw,
        y=y,
        spatial_groups=spatial_groups,
        n_splits=config.number_of_cv_folds,
        threshold=config.classification_threshold,
    )
    cv_summary.to_csv(output_dir / "xgboost_spatial_cross_validation_metrics.csv", index=False)
    oof_metrics.to_csv(output_dir / "xgboost_spatial_oof_metrics.csv", index=False)

    oof_preds_df = df.copy()
    oof_preds_df["spatial_group"] = spatial_groups
    oof_preds_df["xgb_oof_susceptibility_score"] = oof_probs
    oof_preds_df["xgb_oof_predicted_class"] = oof_preds
    oof_preds_df["prediction_type"] = "SPATIAL_OUT_OF_FOLD_VALIDATION"
    oof_preds_df.to_csv(output_dir / "xgboost_spatial_oof_predictions.csv", index=False)

    # 7. Spatial CV Performance Graph
    print("[7/28] Rendering spatial cross-validation error-bar plot...")
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_mean = cv_summary.loc[cv_summary["fold"] == "mean", metric_cols].iloc[0]
    cv_std = cv_summary.loc[cv_summary["fold"] == "std", metric_cols].iloc[0]
    plot_cv_metrics(cv_mean, cv_std, metric_cols, output_dir / "xgboost_spatial_cross_validation.png")
    print(f"       Mean Spatial CV ROC-AUC: {cv_mean['roc_auc']:.4f} (±{cv_std['roc_auc']:.4f})")

    # 8. Spatial Holdout Split (30%)
    print(f"[8/28] Creating {int(config.spatial_holdout_group_fraction*100)}% spatially separated holdout split...")
    h_train_idx, h_test_idx = split_spatial_holdout(
        X_raw,
        y,
        spatial_groups,
        test_size=config.spatial_holdout_group_fraction,
        seed=config.seed,
    )
    h_train_groups = sorted(np.unique(spatial_groups[h_train_idx]).astype(int).tolist())
    h_test_groups = sorted(np.unique(spatial_groups[h_test_idx]).astype(int).tolist())

    X_h_train, X_h_test = X_raw.iloc[h_train_idx], X_raw.iloc[h_test_idx]
    y_h_train, y_h_test = y.iloc[h_train_idx], y.iloc[h_test_idx]

    holdout_model = pipeline_factory()
    holdout_model, h_metrics, h_probs, h_preds = evaluate_holdout(
        model=holdout_model,
        X_train=X_h_train,
        y_train=y_h_train,
        X_test=X_h_test,
        y_test=y_h_test,
        train_groups=h_train_groups,
        test_groups=h_test_groups,
        threshold=config.classification_threshold,
    )
    h_metrics.to_csv(output_dir / "xgboost_spatial_holdout_metrics.csv", index=False)

    # 9. Spatial Holdout Predictions Table
    print("[9/28] Exporting spatial holdout predictions...")
    h_preds_df = df.iloc[h_test_idx].copy()
    h_preds_df["spatial_group"] = spatial_groups[h_test_idx]
    h_preds_df["xgb_susceptibility_score"] = h_probs
    h_preds_df["xgb_predicted_class"] = h_preds
    h_preds_df["prediction_correct"] = (y_h_test.to_numpy() == h_preds)
    h_preds_df["prediction_type"] = "SPATIAL_GROUP_HOLDOUT"
    h_preds_df.to_csv(output_dir / "xgboost_spatial_holdout_predictions.csv", index=False)

    # 10. Explained Confusion Matrix & Classification Report
    print("[10/28] Computing explained confusion matrix & metrics...")
    cm, cm_df, explained_metrics, metrics_dict, report_df = compute_explained_confusion_matrix(
        y_true=y_h_test,
        y_pred=h_preds,
        y_prob=h_probs,
    )
    cm_df.to_csv(output_dir / "xgboost_spatial_holdout_confusion_matrix.csv")
    explained_metrics.to_csv(output_dir / "xgboost_spatial_holdout_explained_metrics.csv", index=False)
    report_df.to_csv(output_dir / "xgboost_spatial_holdout_classification_report.csv")

    plot_explained_confusion_matrix(
        confusion_matrix_data=cm,
        metrics_dict=metrics_dict,
        output_path=output_dir / "xgboost_spatial_holdout_confusion_matrix_explained.png",
    )

    # 11. Spatial Holdout ROC Curve
    print("[11/28] Generating spatial holdout ROC curve...")
    fpr, tpr, roc_df = compute_roc_curve_data(y_h_test, h_probs)
    roc_df.to_csv(output_dir / "xgboost_spatial_holdout_roc_curve.csv", index=False)
    plot_roc_curve(fpr, tpr, metrics_dict["auc"], output_dir / "xgboost_spatial_holdout_roc_curve.png")

    # 12. Save Spatial Holdout Model Pipeline
    print("[12/28] Saving spatial holdout model pipeline (.joblib)...")
    import joblib
    joblib.dump(holdout_model, output_dir / "xgboost_kwri_spatial_holdout_pipeline.joblib")

    # 13. Fit Final Mapping Pipeline on All Points
    print("[13/28] Fitting final deployment pipeline on all points...")
    final_pipeline = pipeline_factory()
    final_pipeline.fit(X_raw, y)
    final_pipeline_path = output_dir / "xgboost_kwri_spatial_final_pipeline.joblib"
    joblib.dump(final_pipeline, final_pipeline_path)

    # 14. In-Sample Fitted Predictions
    print("[14/28] Generating in-sample fitted predictions...")
    in_sample_probs = final_pipeline.predict_proba(X_raw)[:, 1]
    in_sample_preds = (in_sample_probs >= config.classification_threshold).astype(int)
    in_sample_df = df.copy()
    in_sample_df["xgb_in_sample_susceptibility_score"] = in_sample_probs
    in_sample_df["xgb_in_sample_predicted_class"] = in_sample_preds
    in_sample_df["prediction_type"] = "IN_SAMPLE_NOT_VALIDATION"
    in_sample_df.to_csv(output_dir / "xgboost_all_point_predictions_IN_SAMPLE.csv", index=False)

    # 15. Detailed & Grouped Feature Importance
    print("[15/28] Calculating detailed and grouped feature importances...")
    detailed_imp = compute_feature_importance(final_pipeline, config)
    detailed_imp.to_csv(output_dir / "xgboost_final_feature_importance_detailed.csv", index=False)

    grouped_imp = aggregate_landcover_importance(detailed_imp)
    grouped_imp.to_csv(output_dir / "xgboost_final_feature_importance_grouped.csv", index=False)

    # 16. Shared-Scale Importance Plots
    print("[16/28] Rendering feature importance charts (300 DPI)...")
    max_imp = max(detailed_imp["importance"].max(), grouped_imp["importance"].max())
    common_axis_max = min(1.0, np.ceil((max_imp + 0.05) * 10) / 10)

    # Detailed plot
    plot_det = detailed_imp[detailed_imp["importance"] > 0].sort_values("importance", ascending=True)
    det_colors = ["#C45A3D" if lc else "#326273" for lc in plot_det["is_landcover"]]
    det_height = max(7.0, 1.5 + len(plot_det) * 0.43)
    plot_feature_importance(
        df=plot_det,
        feature_col="display_feature",
        importance_col="importance",
        colors=det_colors,
        common_axis_max=common_axis_max,
        title="Final XGBoost Predictor Importance",
        output_path=output_dir / "xgboost_final_feature_importance_detailed.png",
        fig_height=det_height,
    )

    # Grouped plot
    plot_grp = grouped_imp[grouped_imp["importance"] > 0].sort_values("importance", ascending=True)
    grp_colors = ["#C45A3D" if f == "Land Cover Total" else "#326273" for f in plot_grp["grouped_feature"]]
    plot_feature_importance(
        df=plot_grp,
        feature_col="grouped_feature",
        importance_col="importance",
        colors=grp_colors,
        common_axis_max=common_axis_max,
        title="Grouped XGBoost Predictor Importance",
        output_path=output_dir / "xgboost_final_feature_importance_grouped.png",
        fig_height=7.0,
    )

    # 17. Save Final Model Bundle & Run Metadata
    print("[17/28] Serializing full model bundle & run settings JSON...")
    bundle_path = output_dir / "xgboost_kwri_spatial_model_bundle.joblib"
    save_model_bundle(final_pipeline, config, scale_pos_weight, bundle_path)

    stats = {
        "n_records": len(df),
        "class_counts": {"non_landslide": neg_count, "landslide": pos_count},
        "scale_pos_weight": scale_pos_weight,
        "holdout_train_groups": h_train_groups,
        "holdout_test_groups": h_test_groups,
        "mapping_pipeline_path": str(final_pipeline_path.resolve()),
        "model_bundle_path": str(bundle_path.resolve()),
    }
    export_run_metadata(config, stats, output_dir / "xgboost_spatial_run_settings.json")

    # 18. Output Inventory & Summary Dashboard
    print("[18/28] Finalizing output file inventory...")
    inventory = build_results_inventory(output_dir)
    inventory.to_csv(output_dir / "xgboost_spatial_results_inventory.csv", index=False)

    print("\n" + "=" * 75)
    print("KWRI SPATIAL XGBOOST WORKFLOW EXECUTION COMPLETED")
    print("=" * 75)
    print(f"Total Artifacts Created: {len(inventory)} files in {output_dir.resolve()}")
    print("\nPERFORMANCE SUMMARY:")
    print(f"  • Spatial CV Mean ROC-AUC:   {cv_mean['roc_auc']:.4f} (±{cv_std['roc_auc']:.4f})")
    print(f"  • Spatial CV Mean F1-Score:  {cv_mean['f1']:.4f} (±{cv_std['f1']:.4f})")
    print(f"  • Spatial Holdout ROC-AUC:   {metrics_dict['auc']:.4f}")
    print(f"  • Spatial Holdout Accuracy:  {metrics_dict['accuracy']:.4f}")
    print(f"  • Spatial Holdout Precision: {metrics_dict['precision']:.4f}")
    print(f"  • Spatial Holdout Recall:    {metrics_dict['recall']:.4f}")
    print(f"  • Spatial Holdout Specificity: {metrics_dict['specificity']:.4f}")
    print(f"  • Spatial Holdout F1-Score:  {metrics_dict['f1']:.4f}")
    print("\nTOP 5 GROUPED PREDICTORS:")
    for _, row in grouped_imp.head(5).iterrows():
        print(f"  {int(row['rank'])}. {row['grouped_feature']}: {row['importance_percent']:.2f}%")
    print("=" * 75 + "\n")

    if not headless and sys.platform == "win32":
        try:
            open_folder = input("Open output folder in File Explorer? [Y/n]: ").strip().lower()
            if open_folder in ("", "y", "yes"):
                os.startfile(output_dir)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="KWRI Spatial XGBoost Landslide Susceptibility Workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input points CSV")
    parser.add_argument("--output", "-o", type=str, default="outputs/XGBoost_KWRI_Spatial_Final", help="Output directory")
    parser.add_argument("--clusters", type=int, default=10, help="Number of spatial K-Means clusters")
    parser.add_argument("--folds", type=int, default=5, help="Number of spatial CV folds")
    parser.add_argument("--holdout-frac", type=float, default=0.30, help="Spatial holdout fraction")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification probability threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--headless", action="store_true", help="Run without GUI or confirmation prompts")

    args = parser.parse_args()

    # Determine input CSV
    if args.input:
        input_csv = Path(args.input)
    elif args.headless:
        # Check standard default paths
        default_csv = ROOT_DIR / "data" / "sample_test_points.csv"
        if default_csv.exists():
            input_csv = default_csv
        else:
            print("[!] Error: In headless mode, --input must be explicitly provided.")
            sys.exit(1)
    else:
        input_csv = prompt_user_for_input()

    config = WorkflowConfig(
        seed=args.seed,
        number_of_spatial_groups=args.clusters,
        number_of_cv_folds=args.folds,
        spatial_holdout_group_fraction=args.holdout_frac,
        classification_threshold=args.threshold,
        input_csv=input_csv,
        output_dir=Path(args.output),
    )

    run_pipeline(config, headless=args.headless)


if __name__ == "__main__":
    main()
