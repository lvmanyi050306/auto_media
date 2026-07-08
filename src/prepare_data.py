# -*- coding: utf-8 -*-
"""Validate or prepare the Data/<label> folders used by the course pipeline."""

from __future__ import annotations

import argparse

from crop_pages import DEFAULT_CONFIG, crop_all, load_config, rebuild_index_from_existing
from utils import DATA_DIR, LABELS, list_data_images


def validate_data() -> dict[str, int]:
    rows = list_data_images(DATA_DIR)
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    missing = [label for label in LABELS if counts.get(label, 0) == 0]
    if missing:
        raise RuntimeError(f"Data 目录缺少裁切图片: {', '.join(missing)}")
    rebuild_index_from_existing()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", action="store_true", help="Run crop_pages with the default config.")
    parser.add_argument("--force", action="store_true", help="Force recropping if --crop is used.")
    args = parser.parse_args()
    if args.crop:
        crop_all(load_config(DEFAULT_CONFIG), force=args.force)
    counts = validate_data()
    print("Data validation:")
    for label, count in counts.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
