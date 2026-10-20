"""
Kavrepalanchowk Landslide Susceptibility Workflow Runner
========================================================

Unified CLI and Orchestration Entrypoint for the Kavrepalanchowk Landslide
Susceptibility Mapping and Revision Pipeline (MDPI Remote Sensing).

Usage:
    python run_workflow.py --stage audit
    python run_workflow.py --stage spatial
    python run_workflow.py --stage nested --repeats 3 --iterations 20
    python run_workflow.py --stage summarize
    python run_workflow.py --stage final-fit
    python run_workflow.py --stage figures
    python run_workflow.py --stage manuscript
    python run_workflow.py --stage all
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

BANNER = r"""
================================================================================
     KAVREPALANCHOWK LANDSLIDE SUSCEPTIBILITY MAPPING & AUDIT PIPELINE
     MDPI Remote Sensing Revision Workflow - Nepal Middle Hills Study Area
================================================================================
"""


def run_cmd(cmd, env_vars=None):
    """Run command with live output streaming."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    print(f"\n[RUNNING] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, env=env, shell=isinstance(cmd, str))
    if res.returncode != 0:
        print(f"[ERROR] Command exited with code {res.returncode}")
        return False
    return True


def stage_audit():
    print("\n--- STAGE: Data Quality, Coordinates & Geometric Audit ---")
    return run_cmd([sys.executable, "audit.py"])


def stage_spatial(exclude="MBI,EVI,MNDWI,Curvature", tag="_rx2"):
    print("\n--- STAGE: Spatial Autocorrelation & Validation Scheme Sensitivity ---")
    env = {"LS_EXCLUDE": exclude, "LS_TAG": tag}
    return run_cmd([sys.executable, "s1_spatial.py"], env_vars=env)


def stage_nested(repeats=3, iterations=20, exclude="MBI,EVI,MNDWI,Curvature", tag="_rx2", data=None):
    print(f"\n--- STAGE: Nested Spatial Cross-Validation (Repeats={repeats}, Iter={iterations}) ---")
    env = {"LS_EXCLUDE": exclude, "LS_TAG": tag}
    if data:
        env["LS_DATA"] = data
    return run_cmd([sys.executable, "s2_nested.py", str(repeats), str(iterations)], env_vars=env)


def stage_summarize(exclude="MBI,EVI,MNDWI,Curvature", tag="_rx2"):
    print("\n--- STAGE: Metrics Summarization & DeLong Statistical Testing ---")
    env = {"LS_EXCLUDE": exclude, "LS_TAG": tag}
    return run_cmd([sys.executable, "s4_summarize.py"], env_vars=env)


def stage_final_fit(exclude="MBI,EVI,MNDWI,Curvature", tag="_rx2"):
    print("\n--- STAGE: Locked Pipeline Refit on Full Inventory ---")
    env = {"LS_EXCLUDE": exclude, "LS_TAG": tag}
    return run_cmd([sys.executable, "s5_final_fit.py"], env_vars=env)


def stage_figures():
    print("\n--- STAGE: Publication Figures Generation (600 DPI) ---")
    os.makedirs("figs", exist_ok=True)
    scripts = ["fig_study_area.py", "figs1.py", "figs2.py", "fig_workflow.py"]
    success = True
    for s in scripts:
        if os.path.exists(s):
            ok = run_cmd([sys.executable, s])
            if not ok:
                success = False
    return success


def stage_manuscript():
    print("\n--- STAGE: Manuscript, Response Letter & Supplementary Tables Build ---")
    scripts = ["mk_supp.py", "build_ms.py", "build_response.py", "build_cover.py"]
    success = True
    for s in scripts:
        if os.path.exists(s):
            ok = run_cmd([sys.executable, s])
            if not ok:
                success = False
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Unified runner for Kavrepalanchowk Landslide Susceptibility Mapping."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "audit", "spatial", "nested", "summarize", "final-fit", "figures", "manuscript"],
        default="audit",
        help="Pipeline stage to execute (default: audit)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="samples_proj.csv",
        help="Input CSV path (default: samples_proj.csv)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="MBI,EVI,MNDWI,Curvature",
        help="Comma-separated predictors to drop (Model T default: MBI,EVI,MNDWI,Curvature)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="_rx2",
        help="Output file suffix tag (default: _rx2)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of outer spatial CV repeats (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of random search hyperparameter draws (default: 20)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without interactive prompts",
    )

    args = parser.parse_args()
    print(BANNER)

    if args.stage == "audit":
        stage_audit()
    elif args.stage == "spatial":
        stage_spatial(args.exclude, args.tag)
    elif args.stage == "nested":
        stage_nested(args.repeats, args.iterations, args.exclude, args.tag, args.data)
    elif args.stage == "summarize":
        stage_summarize(args.exclude, args.tag)
    elif args.stage == "final-fit":
        stage_final_fit(args.exclude, args.tag)
    elif args.stage == "figures":
        stage_figures()
    elif args.stage == "manuscript":
        stage_manuscript()
    elif args.stage == "all":
        stage_audit()
        stage_spatial(args.exclude, args.tag)
        stage_nested(args.repeats, args.iterations, args.exclude, args.tag, args.data)
        stage_summarize(args.exclude, args.tag)
        stage_final_fit(args.exclude, args.tag)
        stage_figures()
        stage_manuscript()


if __name__ == "__main__":
    main()
