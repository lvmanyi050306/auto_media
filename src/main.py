# -*- coding: utf-8 -*-
"""Command-line entry for Topic 4 and Topic 5."""

from __future__ import annotations

import argparse
import subprocess
import sys

from prepare_data import validate_data
from train_models import run_topic4, run_topic5
from utils import CASE_DIR, PROJECT_ROOT, add_local_pydeps, ensure_output_dirs, model_inventory_rows


def print_model_inventory() -> None:
    print("Local model inventory:")
    for row in model_inventory_rows():
        print(f"  {row['model']}: {row['status']} | {row['size']} | {row['path']}")


def run_case_resnet_if_needed() -> None:
    pca = CASE_DIR / "2_视觉主成分" / "resnet50_PCA_features_part0001.csv"
    if pca.exists():
        return
    script = CASE_DIR / "视觉分析_选题4_ResNet50.py"
    if not script.exists():
        raise FileNotFoundError(f"缺少 ResNet50 兼容分析脚本: {script}")
    env = None
    dep = CASE_DIR / ".pydeps"
    if dep.exists():
        import os

        env = os.environ.copy()
        env["PYTHONPATH"] = str(dep)
    subprocess.run([sys.executable, str(script)], cwd=str(CASE_DIR), check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="智能媒体工作坊 Topic 4/5 pipeline")
    parser.add_argument("--task", choices=["topic4", "topic5", "all", "check"], default="all")
    parser.add_argument("--all", action="store_true", help="Alias for --task all.")
    parser.add_argument("--skip-feature", action="store_true", help="Reuse existing PCA/features only.")
    args = parser.parse_args()
    if args.all:
        args.task = "all"

    add_local_pydeps()
    ensure_output_dirs()
    print(f"Project: {PROJECT_ROOT}")
    print_model_inventory()
    counts = validate_data()
    print("Data counts:", counts)

    if args.task == "check":
        return
    if not args.skip_feature and args.task in {"topic4", "all"}:
        run_case_resnet_if_needed()
    if args.task in {"topic4", "all"}:
        run_topic4()
    if args.task in {"topic5", "all"}:
        run_topic5()
    print("Done. See outputs/topic4 and outputs/topic5.")


if __name__ == "__main__":
    main()
