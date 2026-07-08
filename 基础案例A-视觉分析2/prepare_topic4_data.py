# -*- coding: utf-8 -*-
"""Prepare Topic 4 image groups for 视觉分析.py.

The course visual pipeline treats each subfolder under Data/ as one group.
This script crops copied-symbol cells from the three scanned datasets and
places them into:

Data/端正易读
Data/美观
Data/极致扭曲

Only copied cells are exported; the reference column, page headers, row
numbers, and scan margins are excluded.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "Data"

DATASETS = [
    {"prefix": "3-", "group": "端正易读"},
    {"prefix": "4-", "group": "美观"},
    {"prefix": "5-", "group": "极致扭曲"},
]

# Calibrated for the provided 1536 x 2184 scanned pages.
LEFT_X = [56, 228, 406, 580]
RIGHT_X = [810, 982, 1158, 1332]
Y0 = 132
ROW_PITCH = 161
CELL_W = 140
CELL_H = 146
ROWS_PER_PANEL = 12
COPY_COLS = [1, 2, 3]


def find_dataset_dir(prefix: str) -> Path:
    matches = sorted(
        path for path in WORKSPACE.iterdir() if path.is_dir() and path.name.startswith(prefix)
    )
    if not matches:
        raise FileNotFoundError(f"Cannot find source dataset folder with prefix {prefix!r}")
    return matches[0]


def find_scan_dir(dataset_dir: Path) -> Path:
    matches = sorted(
        path for path in dataset_dir.iterdir() if path.is_dir() and "40" in path.name
    )
    if not matches:
        raise FileNotFoundError(f"Cannot find scan folder under {dataset_dir}")
    return matches[0]


def safe_clear_group_dir(group_dir: Path) -> None:
    """Clear generated files only inside Data/<group>."""
    group_dir.mkdir(parents=True, exist_ok=True)
    resolved_group = group_dir.resolve()
    resolved_data = DATA_DIR.resolve()
    if resolved_data not in resolved_group.parents:
        raise RuntimeError(f"Refusing to clear unexpected path: {resolved_group}")
    for child in group_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def crop_cell(page: Image.Image, x: int, y: int) -> Image.Image:
    crop = page.crop((x, y, x + CELL_W, y + CELL_H)).convert("RGB")
    # Standardize to a square canvas for ViT/CLIP preprocessing stability.
    canvas = Image.new("RGB", (160, 160), "white")
    canvas.paste(crop, ((160 - CELL_W) // 2, (160 - CELL_H) // 2))
    return canvas


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "topic4_manifest.csv"
    rows: list[dict[str, str | int]] = []

    for dataset in DATASETS:
        dataset_dir = find_dataset_dir(dataset["prefix"])
        scan_dir = find_scan_dir(dataset_dir)
        group_dir = DATA_DIR / dataset["group"]
        safe_clear_group_dir(group_dir)

        count = 0
        for page_idx, page_path in enumerate(sorted(scan_dir.glob("*.png")), start=1):
            page = Image.open(page_path).convert("RGB")
            for panel_name, x_positions in (("L", LEFT_X), ("R", RIGHT_X)):
                for row_idx in range(ROWS_PER_PANEL):
                    y = Y0 + row_idx * ROW_PITCH
                    for rep_idx, col_idx in enumerate(COPY_COLS, start=1):
                        image = crop_cell(page, x_positions[col_idx], y)
                        filename = (
                            f"{dataset['group']}_p{page_idx:02d}_{panel_name}"
                            f"_r{row_idx + 1:02d}_rep{rep_idx}.png"
                        )
                        out_path = group_dir / filename
                        image.save(out_path)
                        count += 1
                        rows.append(
                            {
                                "group": dataset["group"],
                                "output": str(out_path.relative_to(DATA_DIR)),
                                "source_file": page_path.name,
                                "page": page_idx,
                                "panel": panel_name,
                                "row_in_panel": row_idx + 1,
                                "replicate": rep_idx,
                            }
                        )
        print(f"{dataset['group']}: {count} images")

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "group",
            "output",
            "source_file",
            "page",
            "panel",
            "row_in_panel",
            "replicate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
