# -*- coding: utf-8 -*-
"""Analyze handwriting deformation across three copying instructions.

The script is intentionally self-contained: it reads the three scanned-page
folders in the workspace, crops the copied symbol cells with a fixed grid,
extracts morphology and reference-distance features, runs statistical tests,
and saves figures/tables for the technical report.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
MIN_VALID_INK_RATIO = 0.005

REPORT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPORT_ROOT.parent
PROCESSED_DIR = REPORT_ROOT / "data" / "processed"
RESULTS_DIR = REPORT_ROOT / "results"

for directory in (PROCESSED_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


DATASETS = [
    {"prefix": "3-", "style": "legible", "label": "端正易读"},
    {"prefix": "4-", "style": "aesthetic", "label": "美观"},
    {"prefix": "5-", "style": "distorted", "label": "极致扭曲"},
]

# Calibrated for the 1536 x 2184 scanned pages. Red columns are references;
# green columns are the three copied attempts per symbol row.
LEFT_X = [56, 228, 406, 580]
RIGHT_X = [810, 982, 1158, 1332]
Y0 = 132
ROW_PITCH = 161
CELL_W = 140
CELL_H = 146
ROWS_PER_PANEL = 12
REFERENCE_COL = 0
COPY_COLS = [1, 2, 3]

FEATURES = [
    "ink_ratio",
    "bbox_width_ratio",
    "bbox_height_ratio",
    "bbox_area_ratio",
    "aspect_ratio",
    "fill_ratio",
    "component_count",
    "hole_count",
    "perimeter_norm",
    "complexity",
    "solidity",
    "elongation",
    "orientation_entropy",
    "edge_density",
    "centroid_x",
    "centroid_y",
    "ref_iou",
    "ref_cosine",
    "ref_chamfer",
]

PLOT_FEATURES = [
    "ink_ratio",
    "bbox_area_ratio",
    "complexity",
    "elongation",
    "orientation_entropy",
    "ref_chamfer",
]

STYLE_ORDER = ["legible", "aesthetic", "distorted"]
STYLE_LABEL = {
    "legible": "端正易读",
    "aesthetic": "美观",
    "distorted": "极致扭曲",
}
STYLE_COLOR = {
    "legible": "#3B82F6",
    "aesthetic": "#10B981",
    "distorted": "#EF4444",
}
PAIR_LABELS = [
    ("legible", "aesthetic"),
    ("legible", "distorted"),
    ("aesthetic", "distorted"),
]


def find_dataset_dir(prefix: str) -> Path:
    matches = sorted(p for p in WORKSPACE.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"Cannot find dataset folder with prefix {prefix!r}")
    return matches[0]


def find_scan_dir(dataset_dir: Path) -> Path:
    matches = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and "40" in p.name)
    if not matches:
        raise FileNotFoundError(f"Cannot find scan folder under {dataset_dir}")
    return matches[0]


def crop_cell(page: Image.Image, x: int, y: int) -> Image.Image:
    return page.crop((x, y, x + CELL_W, y + CELL_H))


def clean_mask(gray: np.ndarray) -> np.ndarray:
    """Convert a cropped cell to an ink mask and remove tiny scan artifacts."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_threshold, _ = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    threshold = max(145, min(218, float(otsu_threshold)))
    mask = (blur < threshold).astype(np.uint8)

    # Keep real strokes while discarding small specks and most faint grid noise.
    n_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label_idx in range(1, n_labels):
        area = stats_cc[label_idx, cv2.CC_STAT_AREA]
        if area >= 8:
            cleaned[labels == label_idx] = 1
    return cleaned


def center_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask.copy()
    cx = float(xs.mean())
    cy = float(ys.mean())
    target_x = (mask.shape[1] - 1) / 2.0
    target_y = (mask.shape[0] - 1) / 2.0
    matrix = np.float32([[1, 0, target_x - cx], [0, 1, target_y - cy]])
    shifted = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return (shifted > 0).astype(np.uint8)


def orientation_entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 5:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) + math.pi) % math.pi
    active = (mask > 0) & (magnitude > 5)
    if not np.any(active):
        return 0.0
    hist, _ = np.histogram(angle[active], bins=12, range=(0, math.pi), weights=magnitude[active])
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum() / math.log2(12))


def extract_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    h, w = mask.shape
    area = int(mask.sum())
    out = {name: 0.0 for name in FEATURES if not name.startswith("ref_")}
    if area == 0:
        return out

    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    bbox_area = bbox_w * bbox_h

    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    contour_area = float(sum(cv2.contourArea(c) for c in contours))
    if contour_area <= 0:
        contour_area = float(area)

    hull_area = 0.0
    if contours:
        points = np.vstack(contours)
        if len(points) >= 3:
            hull_area = float(cv2.contourArea(cv2.convexHull(points)))

    hole_count = 0
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        hole_count = int(sum(1 for item in hierarchy if item[3] != -1))

    n_labels, _, stats_cc, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    component_count = int(sum(1 for i in range(1, n_labels) if stats_cc[i, cv2.CC_STAT_AREA] >= 8))

    coords = np.column_stack((xs.astype(float), ys.astype(float)))
    if len(coords) >= 3:
        cov = np.cov(coords, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        elongation = float(math.sqrt(eigvals.max() / eigvals.min()))
    else:
        elongation = 1.0

    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)

    out.update(
        {
            "ink_ratio": float(area / (h * w)),
            "bbox_width_ratio": float(bbox_w / w),
            "bbox_height_ratio": float(bbox_h / h),
            "bbox_area_ratio": float(bbox_area / (h * w)),
            "aspect_ratio": float(bbox_w / max(bbox_h, 1)),
            "fill_ratio": float(area / max(bbox_area, 1)),
            "component_count": float(component_count),
            "hole_count": float(hole_count),
            "perimeter_norm": float(perimeter / math.sqrt(h * w)),
            "complexity": float((perimeter * perimeter) / max(4 * math.pi * area, 1e-6)),
            "solidity": float(area / max(hull_area, area, 1.0)),
            "elongation": elongation,
            "orientation_entropy": orientation_entropy(gray, mask),
            "edge_density": float((edges > 0).sum() / (h * w)),
            "centroid_x": float(xs.mean() / w),
            "centroid_y": float(ys.mean() / h),
        }
    )
    return out


def reference_features(copy_mask: np.ndarray, ref_mask: np.ndarray) -> dict[str, float]:
    copy = center_mask(copy_mask)
    ref = center_mask(ref_mask)
    copy_sum = int(copy.sum())
    ref_sum = int(ref.sum())
    if copy_sum == 0 or ref_sum == 0:
        return {"ref_iou": 0.0, "ref_cosine": 0.0, "ref_chamfer": 1.0}

    intersection = int(np.logical_and(copy, ref).sum())
    union = int(np.logical_or(copy, ref).sum())
    iou = intersection / max(union, 1)
    cosine = intersection / math.sqrt(copy_sum * ref_sum)

    dist_to_ref = cv2.distanceTransform((1 - ref).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_copy = cv2.distanceTransform((1 - copy).astype(np.uint8), cv2.DIST_L2, 3)
    copy_to_ref = float(dist_to_ref[copy > 0].mean()) if copy_sum else 0.0
    ref_to_copy = float(dist_to_copy[ref > 0].mean()) if ref_sum else 0.0
    diag = math.sqrt(copy.shape[0] * copy.shape[0] + copy.shape[1] * copy.shape[1])
    chamfer = (copy_to_ref + ref_to_copy) / (2 * diag)
    return {"ref_iou": float(iou), "ref_cosine": float(cosine), "ref_chamfer": float(chamfer)}


def extract_all_samples() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        dataset_dir = find_dataset_dir(dataset["prefix"])
        scan_dir = find_scan_dir(dataset_dir)
        page_paths = sorted(scan_dir.glob("*.png"))
        if not page_paths:
            raise FileNotFoundError(f"No PNG files in {scan_dir}")
        for page_idx, page_path in enumerate(page_paths, start=1):
            page = Image.open(page_path).convert("L")
            for panel_name, x_positions in (("L", LEFT_X), ("R", RIGHT_X)):
                for row_idx in range(ROWS_PER_PANEL):
                    y = Y0 + row_idx * ROW_PITCH
                    ref_crop = crop_cell(page, x_positions[REFERENCE_COL], y)
                    ref_gray = np.array(ref_crop)
                    ref_mask = clean_mask(ref_gray)
                    for rep_idx, col_idx in enumerate(COPY_COLS, start=1):
                        crop = crop_cell(page, x_positions[col_idx], y)
                        gray = np.array(crop)
                        mask = clean_mask(gray)
                        features = extract_features(gray, mask)
                        features.update(reference_features(mask, ref_mask))
                        row: dict[str, object] = {
                            "style": dataset["style"],
                            "style_label": dataset["label"],
                            "page": page_idx,
                            "panel": panel_name,
                            "row_in_panel": row_idx + 1,
                            "replicate": rep_idx,
                            "source_file": page_path.name,
                        }
                        row.update(features)
                        rows.append(row)
    return rows


def write_feature_csv(rows: list[dict[str, object]], filename: str) -> Path:
    out_path = PROCESSED_DIR / filename
    fieldnames = [
        "style",
        "style_label",
        "is_valid_sample",
        "page",
        "panel",
        "row_in_panel",
        "replicate",
        "source_file",
    ] + FEATURES
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out_path


def mark_valid_samples(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in rows:
        row["is_valid_sample"] = int(float(row["ink_ratio"]) >= MIN_VALID_INK_RATIO)
    return rows


def write_sample_counts(rows: list[dict[str, object]]) -> Path:
    out_path = RESULTS_DIR / "sample_counts.csv"
    counts: dict[str, dict[str, int]] = {
        style: {"total": 0, "valid": 0} for style in STYLE_ORDER
    }
    for row in rows:
        style = str(row["style"])
        counts[style]["total"] += 1
        counts[style]["valid"] += int(row["is_valid_sample"])

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "style",
                "style_label",
                "total_samples",
                "valid_samples",
                "excluded_samples",
                "valid_rate",
            ],
        )
        writer.writeheader()
        for style in STYLE_ORDER:
            total = counts[style]["total"]
            valid = counts[style]["valid"]
            writer.writerow(
                {
                    "style": style,
                    "style_label": STYLE_LABEL[style],
                    "total_samples": total,
                    "valid_samples": valid,
                    "excluded_samples": total - valid,
                    "valid_rate": f"{valid / total:.4f}" if total else "0.0000",
                }
            )
    return out_path


def values_by_style(rows: list[dict[str, object]], feature: str) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[feature])
        if math.isfinite(value):
            grouped[str(row["style"])].append(value)
    return {style: np.array(values, dtype=float) for style, values in grouped.items()}


def cliffs_delta_from_u(u_value: float, n1: int, n2: int) -> float:
    return float((2.0 * u_value) / (n1 * n2) - 1.0)


def run_statistics(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    summary_path = RESULTS_DIR / "summary_by_style.csv"
    tests_path = RESULTS_DIR / "statistical_tests.csv"

    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["feature", "style", "style_label", "n", "mean", "std", "median", "iqr", "min", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feature in FEATURES:
            grouped = values_by_style(rows, feature)
            for style in STYLE_ORDER:
                arr = grouped.get(style, np.array([], dtype=float))
                if len(arr) == 0:
                    continue
                writer.writerow(
                    {
                        "feature": feature,
                        "style": style,
                        "style_label": STYLE_LABEL[style],
                        "n": len(arr),
                        "mean": f"{arr.mean():.6f}",
                        "std": f"{arr.std(ddof=1):.6f}",
                        "median": f"{np.median(arr):.6f}",
                        "iqr": f"{np.quantile(arr, 0.75) - np.quantile(arr, 0.25):.6f}",
                        "min": f"{arr.min():.6f}",
                        "max": f"{arr.max():.6f}",
                    }
                )

    with tests_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "feature",
            "kruskal_H",
            "kruskal_p",
            "pair",
            "mannwhitney_U",
            "pair_p",
            "cliffs_delta",
            "direction",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feature in FEATURES:
            grouped = values_by_style(rows, feature)
            arrays = [grouped[style] for style in STYLE_ORDER if style in grouped]
            h_stat, p_value = stats.kruskal(*arrays)
            for left, right in PAIR_LABELS:
                arr_l = grouped[left]
                arr_r = grouped[right]
                u_value, pair_p = stats.mannwhitneyu(arr_l, arr_r, alternative="two-sided")
                delta = cliffs_delta_from_u(float(u_value), len(arr_l), len(arr_r))
                writer.writerow(
                    {
                        "feature": feature,
                        "kruskal_H": f"{h_stat:.6f}",
                        "kruskal_p": f"{p_value:.6e}",
                        "pair": f"{STYLE_LABEL[left]} vs {STYLE_LABEL[right]}",
                        "mannwhitney_U": f"{u_value:.3f}",
                        "pair_p": f"{pair_p:.6e}",
                        "cliffs_delta": f"{delta:.6f}",
                        "direction": STYLE_LABEL[left] if delta > 0 else STYLE_LABEL[right],
                    }
                )
    return summary_path, tests_path


def feature_matrix(rows: list[dict[str, object]], selected: list[str] | None = None):
    selected_features = selected or FEATURES
    x = np.array([[float(row[name]) for name in selected_features] for row in rows], dtype=float)
    y = np.array([str(row["style"]) for row in rows])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y, selected_features


def run_classification(rows: list[dict[str, object]]) -> tuple[Path, Path, Path]:
    # Do not include centroid features in the classifier; they can reflect small crop offsets.
    selected = [f for f in FEATURES if f not in {"centroid_x", "centroid_y"}]
    x, y, selected = feature_matrix(rows, selected)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(x_train_scaled, y_train)
    y_pred = clf.predict(x_test_scaled)

    metrics_path = RESULTS_DIR / "classification_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("Three-style classification\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=[STYLE_LABEL[s] for s in STYLE_ORDER], labels=STYLE_ORDER))

    cm = confusion_matrix(y_test, y_pred, labels=STYLE_ORDER)
    cm_path = RESULTS_DIR / "confusion_matrix_three_style.csv"
    write_matrix_csv(cm_path, cm, STYLE_ORDER, STYLE_ORDER)

    importances = sorted(zip(selected, clf.feature_importances_), key=lambda item: item[1], reverse=True)
    importance_path = RESULTS_DIR / "feature_importance_three_style.csv"
    with importance_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows((name, f"{value:.8f}") for name, value in importances)

    # Binary extension: aesthetic vs distorted, matching the user's topic-5 focus.
    binary_rows = [row for row in rows if row["style"] in {"aesthetic", "distorted"}]
    xb, yb, selected_b = feature_matrix(binary_rows, selected)
    xb_train, xb_test, yb_train, yb_test = train_test_split(
        xb, yb, test_size=0.25, random_state=RANDOM_SEED, stratify=yb
    )
    scaler_b = StandardScaler()
    xb_train_scaled = scaler_b.fit_transform(xb_train)
    xb_test_scaled = scaler_b.transform(xb_test)
    clf_b = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf_b.fit(xb_train_scaled, yb_train)
    yb_pred = clf_b.predict(xb_test_scaled)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write("\n\nAesthetic-vs-distorted classification\n")
        f.write(f"Accuracy: {accuracy_score(yb_test, yb_pred):.4f}\n")
        f.write(f"Macro F1: {f1_score(yb_test, yb_pred, average='macro'):.4f}\n\n")
        f.write(
            classification_report(
                yb_test,
                yb_pred,
                target_names=[STYLE_LABEL[s] for s in ["aesthetic", "distorted"]],
                labels=["aesthetic", "distorted"],
            )
        )

    cm_b = confusion_matrix(yb_test, yb_pred, labels=["aesthetic", "distorted"])
    cm_b_path = RESULTS_DIR / "confusion_matrix_aesthetic_vs_distorted.csv"
    write_matrix_csv(cm_b_path, cm_b, ["aesthetic", "distorted"], ["aesthetic", "distorted"])

    importance_binary_path = RESULTS_DIR / "feature_importance_aesthetic_vs_distorted.csv"
    importances_b = sorted(zip(selected_b, clf_b.feature_importances_), key=lambda item: item[1], reverse=True)
    with importance_binary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows((name, f"{value:.8f}") for name, value in importances_b)

    return metrics_path, cm_path, importance_binary_path


def write_matrix_csv(path: Path, matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted"] + [STYLE_LABEL.get(s, s) for s in col_labels])
        for label, values in zip(row_labels, matrix):
            writer.writerow([STYLE_LABEL.get(label, label)] + list(map(int, values)))


def set_chinese_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_feature_boxplots(rows: list[dict[str, object]]) -> Path:
    set_chinese_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    axes = axes.ravel()
    for ax, feature in zip(axes, PLOT_FEATURES):
        data = [values_by_style(rows, feature)[style] for style in STYLE_ORDER]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, style in zip(bp["boxes"], STYLE_ORDER):
            patch.set(facecolor=STYLE_COLOR[style], alpha=0.45)
        ax.set_title(feature)
        ax.set_xticks([1, 2, 3], [STYLE_LABEL[s] for s in STYLE_ORDER], rotation=0)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("三种书写目标下的关键字迹变形特征分布", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = RESULTS_DIR / "feature_boxplots.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_pca(rows: list[dict[str, object]]) -> Path:
    set_chinese_style()
    x, y, selected = feature_matrix(rows, [f for f in FEATURES if f not in {"centroid_x", "centroid_y"}])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    coords = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(x_scaled)

    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    max_per_style = 1200
    for style in STYLE_ORDER:
        idx = np.where(y == style)[0]
        if len(idx) > max_per_style:
            idx = np.random.default_rng(RANDOM_SEED).choice(idx, size=max_per_style, replace=False)
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=9,
            alpha=0.35,
            color=STYLE_COLOR[style],
            label=STYLE_LABEL[style],
            edgecolors="none",
        )
    ax.set_title("手工形态特征的PCA二维投影")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = RESULTS_DIR / "pca_scatter_features.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm_path: Path) -> Path:
    set_chinese_style()
    with cm_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    labels = rows[0][1:]
    matrix = np.array([[int(x) for x in row[1:]] for row in rows[1:]], dtype=int)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("三分类混淆矩阵")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = RESULTS_DIR / "confusion_matrix_three_style.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_feature_importance(importance_path: Path) -> Path:
    set_chinese_style()
    with importance_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        items = [(row["feature"], float(row["importance"])) for row in reader]
    items = items[:10][::-1]
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(labels, values, color="#0F766E", alpha=0.85)
    ax.set_title("美观 vs 极致扭曲分类的前10个重要特征")
    ax.set_xlabel("Random Forest importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path = RESULTS_DIR / "feature_importance_aesthetic_vs_distorted.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_effect_heatmap(tests_path: Path) -> Path:
    set_chinese_style()
    pair_names = [f"{STYLE_LABEL[a]} vs {STYLE_LABEL[b]}" for a, b in PAIR_LABELS]
    matrix = np.zeros((len(PLOT_FEATURES), len(pair_names)), dtype=float)
    with tests_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["feature"] in PLOT_FEATURES and row["pair"] in pair_names:
                matrix[PLOT_FEATURES.index(row["feature"]), pair_names.index(row["pair"])] = float(row["cliffs_delta"])

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pair_names)), pair_names, rotation=25, ha="right")
    ax.set_yticks(range(len(PLOT_FEATURES)), PLOT_FEATURES)
    ax.set_title("关键特征的Cliff's delta效应量")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = RESULTS_DIR / "effect_size_heatmap.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def create_example_contact_sheet() -> Path:
    """Create a small visual panel showing reference/copy crops in each style."""
    style_pages = []
    for dataset in DATASETS:
        dataset_dir = find_dataset_dir(dataset["prefix"])
        scan_dir = find_scan_dir(dataset_dir)
        page_path = sorted(scan_dir.glob("*.png"))[0]
        page = Image.open(page_path).convert("L")
        style_pages.append((dataset["label"], page))

    pad = 18
    cell_scale = 1
    label_h = 32
    row_h = CELL_H * cell_scale + label_h + pad
    sheet_w = 4 * CELL_W * cell_scale + 5 * pad
    sheet_h = len(style_pages) * row_h + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    y = pad
    # Use row 4 in left panel: enough strokes but not overly dense.
    sample_y = Y0 + 3 * ROW_PITCH
    for label, page in style_pages:
        draw.text((pad, y), label, fill="#111111", font=font)
        x = pad
        for idx, col_x in enumerate(LEFT_X):
            crop = crop_cell(page, col_x, sample_y).convert("RGB")
            sheet.paste(crop, (x, y + label_h))
            border = "#DC2626" if idx == 0 else "#059669"
            draw.rectangle(
                (x, y + label_h, x + CELL_W, y + label_h + CELL_H),
                outline=border,
                width=3,
            )
            caption = "参考" if idx == 0 else f"抄写{idx}"
            draw.text((x + 4, y + label_h + 4), caption, fill=border, font=font)
            x += CELL_W + pad
        y += row_h

    out_path = RESULTS_DIR / "example_crops_contact_sheet.png"
    sheet.save(out_path)
    return out_path


def main() -> None:
    print("Extracting handwriting deformation features...")
    rows = mark_valid_samples(extract_all_samples())
    feature_csv = write_feature_csv(rows, "handwriting_deformation_features_all.csv")
    valid_rows = [row for row in rows if int(row["is_valid_sample"]) == 1]
    valid_csv = write_feature_csv(valid_rows, "handwriting_deformation_features_valid.csv")
    counts_path = write_sample_counts(rows)
    print(f"Saved all features: {feature_csv}")
    print(f"Saved valid features: {valid_csv}")
    print(f"Saved sample counts: {counts_path}")
    print(f"Samples: {len(rows)} total, {len(valid_rows)} valid")

    print("Running statistical tests...")
    summary_path, tests_path = run_statistics(valid_rows)
    print(f"Saved summary: {summary_path}")
    print(f"Saved tests: {tests_path}")

    print("Running classifiers...")
    metrics_path, cm_path, importance_binary_path = run_classification(valid_rows)
    print(f"Saved classification metrics: {metrics_path}")

    print("Rendering figures...")
    create_example_contact_sheet()
    plot_feature_boxplots(valid_rows)
    plot_pca(valid_rows)
    plot_confusion_matrix(cm_path)
    plot_feature_importance(importance_binary_path)
    plot_effect_heatmap(tests_path)
    print(f"Done. Results are in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
