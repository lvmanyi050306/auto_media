# -*- coding: utf-8 -*-
"""Plotting helpers used by Topic 4 and Topic 5."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from utils import add_local_pydeps

add_local_pydeps()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


COLORS = {
    "端正易读": "#2563eb",
    "美观": "#059669",
    "极致扭曲": "#dc2626",
}


def setup_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def save_scatter(df, x_col: str, y_col: str, label_col: str, title: str, out: Path) -> None:
    setup_chinese_font()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for label, part in df.groupby(label_col):
        ax.scatter(
            part[x_col],
            part[y_col],
            s=9,
            alpha=0.58,
            label=label,
            c=COLORS.get(label, None),
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=False, markerscale=2)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def save_confusion_matrix(cm, labels: list[str], title: str, out: Path) -> None:
    setup_chinese_font()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def save_metric_bars(df, out: Path) -> None:
    setup_chinese_font()
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy", color="#2563eb")
    ax.bar(x + width / 2, df["macro_f1"], width, label="Macro F1", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("美观 vs 极致扭曲：不同特征设置的识别性能")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    for i, row in df.iterrows():
        ax.text(i - width / 2, row["accuracy"] + 0.015, f"{row['accuracy']:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, row["macro_f1"] + 0.015, f"{row['macro_f1']:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _font(size: int = 16):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def save_image_sheet(rows, out: Path, title: str, thumb_size=(140, 140), cols: int = 6) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    font = _font(16)
    title_font = _font(22)
    cell_w = thumb_size[0] + 32
    cell_h = thumb_size[1] + 54
    header_h = 46
    grid_rows = int(np.ceil(len(rows) / cols))
    canvas = Image.new("RGB", (cols * cell_w, header_h + grid_rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), title, fill=(20, 20, 20), font=title_font)
    for idx, item in enumerate(rows):
        image_path = Path(item["image_path"])
        if not image_path.exists():
            continue
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(thumb_size)
        x = (idx % cols) * cell_w + 16
        y = header_h + (idx // cols) * cell_h + 6
        frame = Image.new("RGB", thumb_size, "white")
        frame.paste(img, ((thumb_size[0] - img.width) // 2, (thumb_size[1] - img.height) // 2))
        canvas.paste(frame, (x, y))
        label = item.get("caption") or item.get("label") or ""
        draw.text((x, y + thumb_size[1] + 6), str(label)[:18], fill=(40, 40, 40), font=font)
    canvas.save(out, dpi=(300, 300))
