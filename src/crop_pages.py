# -*- coding: utf-8 -*-
"""Crop scan pages into single copied-symbol cells.

The script is conservative by default: if Data/<label> already contains images,
it rebuilds the index and does not overwrite crops. Use --force to regenerate.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image

from utils import DATA_DIR, PROJECT_ROOT, list_data_images


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "crop_config.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_dataset_dir(prefix: str) -> Path:
    matches = sorted(p for p in PROJECT_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"未找到前缀为 {prefix!r} 的原始数据目录")
    return matches[0]


def find_scan_dir(dataset_dir: Path, keyword: str) -> Path:
    matches = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and keyword in p.name)
    if not matches:
        raise FileNotFoundError(f"未在 {dataset_dir} 下找到包含 {keyword!r} 的扫描页目录")
    return matches[0]


def clear_generated_dir(group_dir: Path) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    resolved = group_dir.resolve()
    if DATA_DIR.resolve() not in resolved.parents:
        raise RuntimeError(f"拒绝清空非 Data 子目录: {resolved}")
    for child in group_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def write_index(rows: list[dict[str, object]]) -> Path:
    out = DATA_DIR / "image_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "label", "source_page", "page_id", "crop_id", "row", "col"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out


def rebuild_index_from_existing() -> Path:
    rows = list_data_images(DATA_DIR)
    return write_index(rows)


def crop_cell(page: Image.Image, x: int, y: int, cfg: dict) -> Image.Image:
    w = int(cfg["cell_width"])
    h = int(cfg["cell_height"])
    out_w, out_h = cfg.get("output_size", [w, h])
    crop = page.crop((x, y, x + w, y + h)).convert("RGB")
    canvas = Image.new("RGB", (int(out_w), int(out_h)), "white")
    canvas.paste(crop, ((int(out_w) - w) // 2, (int(out_h) - h) // 2))
    return canvas


def crop_all(cfg: dict, force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = list_data_images(DATA_DIR)
    if existing and not force:
        print(f"检测到已有裁切图片 {len(existing)} 张；未使用 --force，跳过重新裁切。")
        return rebuild_index_from_existing()

    rows: list[dict[str, object]] = []
    for ds in cfg["datasets"]:
        label = ds["label"]
        source_dir = find_dataset_dir(ds["source_prefix"])
        scan_dir = find_scan_dir(source_dir, cfg.get("scan_folder_keyword", "40"))
        out_dir = DATA_DIR / label
        if force:
            clear_generated_dir(out_dir)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        page_paths = sorted(p for p in scan_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for page_no, page_path in enumerate(page_paths, start=1):
            page = Image.open(page_path).convert("RGB")
            panels = (("L", cfg["left_x"]), ("R", cfg["right_x"]))
            for panel, x_positions in panels:
                for row_idx in range(int(cfg["rows_per_panel"])):
                    y = int(cfg["start_y"]) + row_idx * int(cfg["row_pitch"])
                    for rep, col_idx in enumerate(cfg["copy_columns"], start=1):
                        x = int(x_positions[int(col_idx)])
                        image = crop_cell(page, x, y, cfg)
                        filename = f"{label}_p{page_no:02d}_{panel}_r{row_idx + 1:02d}_rep{rep}.png"
                        out_path = out_dir / filename
                        image.save(out_path)
                        rows.append(
                            {
                                "image_path": str(out_path),
                                "label": label,
                                "source_page": page_path.name,
                                "page_id": f"p{page_no:02d}_{panel}",
                                "crop_id": out_path.stem,
                                "row": row_idx + 1,
                                "col": rep,
                            }
                        )
                        count += 1
        print(f"{label}: {count} 张")
    return write_index(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop handwriting scan pages into Data/<label> cells.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="Regenerate crops and overwrite generated Data folders.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = crop_all(cfg, force=args.force)
    print(f"image_index: {index}")


if __name__ == "__main__":
    main()
