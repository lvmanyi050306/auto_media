# -*- coding: utf-8 -*-
"""Shared paths and helpers for the handwriting visual analysis project."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = PROJECT_ROOT / "基础案例A-视觉分析2"
CASE_SCRIPT = CASE_DIR / "视觉分析.py"
DATA_DIR = CASE_DIR / "Data"
MODEL_DIR = PROJECT_ROOT / "视觉特征计算" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TOPIC4_DIR = OUTPUT_DIR / "topic4"
TOPIC5_DIR = OUTPUT_DIR / "topic5"

LABELS = ["端正易读", "美观", "极致扭曲"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def add_local_pydeps() -> None:
    """Prefer the project-local dependency folder when it exists."""
    dep = CASE_DIR / ".pydeps"
    if dep.exists() and str(dep) not in sys.path:
        sys.path.insert(0, str(dep))


def ensure_output_dirs() -> None:
    for topic_dir in (TOPIC4_DIR, TOPIC5_DIR):
        for name in ("figures", "tables", "metrics"):
            (topic_dir / name).mkdir(parents=True, exist_ok=True)


def scan_model_resources() -> dict[str, object]:
    """Return local model availability without downloading anything."""
    torchvision_dir = MODEL_DIR / "torchvision"
    hf_dir = MODEL_DIR / "hf_models"
    return {
        "ResNet50": torchvision_dir / "ResNet50.pth",
        "Swin_T": torchvision_dir / "Swin_T.pth",
        "ViT_B16": torchvision_dir / "ViT_B16.pth",
        "CLIP": hf_dir / "openai__clip-vit-base-patch32",
        "DINOv2": hf_dir / "facebook__dinov2-base",
    }


def model_inventory_rows() -> list[dict[str, str]]:
    rows = []
    for name, path in scan_model_resources().items():
        p = Path(path)
        if p.is_file():
            size = f"{p.stat().st_size / (1024 ** 2):.1f} MB"
            status = "存在"
        elif p.is_dir():
            size = "-"
            status = "存在"
        else:
            size = "-"
            status = "缺失"
        rows.append({"model": name, "path": str(p), "status": status, "size": size})
    return rows


def parse_crop_metadata(filename: str) -> dict[str, object]:
    """Parse names like 美观_p01_L_r02_rep3."""
    stem = Path(filename).stem
    pat = re.compile(r"^(?P<label>.+)_p(?P<page>\d+)_(?P<panel>[LR])_r(?P<row>\d+)_rep(?P<rep>\d+)$")
    match = pat.match(stem)
    if not match:
        return {
            "page_id": "",
            "crop_id": stem,
            "row": "",
            "col": "",
            "panel": "",
            "replicate": "",
        }
    g = match.groupdict()
    page = int(g["page"])
    row = int(g["row"])
    rep = int(g["rep"])
    return {
        "page_id": f"p{page:02d}_{g['panel']}",
        "crop_id": stem,
        "row": row,
        "col": rep,
        "panel": g["panel"],
        "replicate": rep,
    }


def list_data_images(data_dir: Path = DATA_DIR) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not data_dir.exists():
        return rows
    for label_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            meta = parse_crop_metadata(image_path.name)
            rows.append(
                {
                    "image_path": str(image_path),
                    "relative_path": str(image_path.relative_to(PROJECT_ROOT)),
                    "label": label_dir.name,
                    "filename": image_path.stem,
                    "source_page": meta["page_id"],
                    **meta,
                }
            )
    return rows


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
