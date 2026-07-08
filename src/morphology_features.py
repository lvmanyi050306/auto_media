# -*- coding: utf-8 -*-
"""Morphology features for copied symbol cells."""

from __future__ import annotations

from pathlib import Path

from utils import PROJECT_ROOT, add_local_pydeps

add_local_pydeps()

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


RAW_FEATURES = PROJECT_ROOT / "选题5_基于题4拓展分析" / "data" / "processed" / "handwriting_deformation_features_valid.csv"

MORPHOLOGY_FEATURES = [
    "ink_ratio",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "bbox_area_ratio",
    "aspect_ratio",
    "perimeter_norm",
    "complexity",
    "edge_density",
    "centroid_x",
    "centroid_y",
    "direction_entropy",
    "ref_chamfer",
]


def load_existing_morphology() -> pd.DataFrame:
    """Load the existing valid morphology table and normalize naming."""
    if not RAW_FEATURES.exists():
        raise FileNotFoundError(f"未找到已有形态学特征表: {RAW_FEATURES}")
    df = pd.read_csv(RAW_FEATURES)
    df = df[df["style_label"].isin(["美观", "极致扭曲"])].copy()
    df["filename"] = df.apply(
        lambda r: f"{r['style_label']}_p{int(r['page']):02d}_{r['panel']}_r{int(r['row_in_panel']):02d}_rep{int(r['replicate'])}",
        axis=1,
    )
    df["group"] = df["style_label"]
    if "orientation_entropy" in df.columns and "direction_entropy" not in df.columns:
        df["direction_entropy"] = df["orientation_entropy"]
    return df


def extract_basic_morphology(image_path: Path) -> dict[str, float]:
    """Compute a compact, dependency-light feature set from one image."""
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img).astype(np.float32)
    mask = arr < 210
    h, w = mask.shape
    ink = float(mask.mean())
    if not mask.any():
        return {name: 0.0 for name in MORPHOLOGY_FEATURES}

    ys, xs = np.where(mask)
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    bbox_area = bw * bh
    eroded = ndimage.binary_erosion(mask)
    perimeter = float(np.logical_xor(mask, eroded).sum())
    gy, gx = np.gradient(mask.astype(float))
    edge = np.hypot(gx, gy) > 0
    angles = np.arctan2(gy[edge], gx[edge])
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=False)
    prob = hist / max(hist.sum(), 1)
    entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum() / 3.0)
    return {
        "ink_ratio": ink,
        "bbox_width_ratio": float(bw / w),
        "bbox_height_ratio": float(bh / h),
        "bbox_area_ratio": float(bbox_area / (w * h)),
        "aspect_ratio": float(bw / max(bh, 1)),
        "perimeter_norm": float(perimeter / max(w + h, 1)),
        "complexity": float((perimeter ** 2) / max(mask.sum(), 1)),
        "edge_density": float(edge.mean()),
        "centroid_x": float(xs.mean() / w),
        "centroid_y": float(ys.mean() / h),
        "direction_entropy": entropy,
        "ref_chamfer": np.nan,
    }
