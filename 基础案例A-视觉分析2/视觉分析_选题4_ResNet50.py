# -*- coding: utf-8 -*-
"""Topic 4 visual analysis with local ResNet50 features.

This is a CPU-friendly companion to the course 视觉分析.py. It uses the same
Data/<group> convention and writes the same staged outputs, but extracts
features with the locally available torchvision ResNet50 checkpoint.
"""

from __future__ import annotations

import csv
import inspect
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DATA_ROOT = ROOT / "Data"
FEATURE_DIR = ROOT / "1_视觉特征"
PCA_DIR = ROOT / "2_视觉主成分"
DIM2_DIR = ROOT / "3_降维到二维平面"
STATS_DIR = ROOT / "4_选题4统计检验"

for directory in (FEATURE_DIR, PCA_DIR, DIM2_DIR, STATS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_TAG = "resnet50"
BATCH_SIZE = 64
PCA_COMPONENTS = 50
TSNE_PERPLEXITY = 30
TSNE_ITER = 650

GROUP_ORDER = ["端正易读", "美观", "极致扭曲"]
GROUP_COLORS = {
    "端正易读": "#3B82F6",
    "美观": "#10B981",
    "极致扭曲": "#EF4444",
}


def list_all_images() -> pd.DataFrame:
    rows = []
    for group_dir in sorted(DATA_ROOT.iterdir()):
        if not group_dir.is_dir():
            continue
        for image_path in sorted(group_dir.glob("*.png")):
            rows.append(
                {
                    "path": str(image_path.relative_to(ROOT)),
                    "filename": image_path.stem,
                    "group": group_dir.name,
                }
            )
    if not rows:
        raise RuntimeError(f"No PNG images found under {DATA_ROOT}")
    df = pd.DataFrame(rows)
    df.to_csv(FEATURE_DIR / "image_index.csv", index=False, encoding="utf-8-sig")
    return df


def find_resnet_weight() -> Path:
    matches = sorted(WORKSPACE.glob("*/model/torchvision/ResNet50.pth"))
    if not matches:
        matches = sorted(WORKSPACE.glob("*/model/torch/hub/checkpoints/resnet50-*.pth"))
    if not matches:
        raise FileNotFoundError("Cannot find local ResNet50 checkpoint")
    return matches[0]


def load_feature_model():
    weight_path = find_resnet_weight()
    model = models.resnet50(weights=None)
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    print(f"model: {weight_path}")
    print(f"device: {device}")
    return model, preprocess, device


def extract_features(index_df: pd.DataFrame) -> np.ndarray:
    model, preprocess, device = load_feature_model()
    paths = [ROOT / p for p in index_df["path"].tolist()]
    features = []
    total = len(paths)
    print(f"Extracting {MODEL_TAG} features for {total} images...")
    for start in range(0, total, BATCH_SIZE):
        batch_paths = paths[start : start + BATCH_SIZE]
        batch = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        ).to(device)
        with torch.no_grad():
            out = model(batch).flatten(1).cpu().numpy().astype(np.float32)
        features.append(out)
        done = min(start + BATCH_SIZE, total)
        if done % 640 == 0 or done == total:
            print(f"  {done}/{total}")
    return np.vstack(features)


def write_feature_csv(index_df: pd.DataFrame, features: np.ndarray) -> Path:
    out = FEATURE_DIR / f"{MODEL_TAG}_features_part0001.csv"
    df = pd.DataFrame(features, columns=[f"f{i + 1}" for i in range(features.shape[1])])
    df.insert(0, "group", index_df["group"].values)
    df.insert(0, "filename", index_df["filename"].values)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def run_pca(index_df: pd.DataFrame, features: np.ndarray) -> tuple[np.ndarray, Path]:
    x = StandardScaler().fit_transform(features)
    n_components = min(PCA_COMPONENTS, x.shape[1], x.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=SEED)
    pcs = pca.fit_transform(x)

    out = PCA_DIR / f"{MODEL_TAG}_PCA_features_part0001.csv"
    df = pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(pcs.shape[1])])
    df.insert(0, "group", index_df["group"].values)
    df.insert(0, "filename", index_df["filename"].values)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    variance = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    variance.to_csv(PCA_DIR / f"{MODEL_TAG}_PCA_variance.csv", index=False, encoding="utf-8-sig")
    return pcs, out


def run_tsne(index_df: pd.DataFrame, pcs: np.ndarray) -> tuple[np.ndarray, Path]:
    kwargs = {
        "n_components": 2,
        "perplexity": TSNE_PERPLEXITY,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": SEED,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = TSNE_ITER
    else:
        kwargs["n_iter"] = TSNE_ITER
    print("Running t-SNE...")
    coords = TSNE(**kwargs).fit_transform(pcs)
    out = DIM2_DIR / f"{MODEL_TAG}_tSNE.csv"
    df = pd.DataFrame(
        {
            "filename": index_df["filename"].values,
            "group": index_df["group"].values,
            "TSNE_X": coords[:, 0],
            "TSNE_Y": coords[:, 1],
        }
    )
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return coords, out


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_tsne(index_df: pd.DataFrame, coords: np.ndarray) -> Path:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 7.6))
    groups = index_df["group"].values
    for group in GROUP_ORDER:
        mask = groups == group
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=0.45,
            c=GROUP_COLORS[group],
            label=group,
            edgecolors="none",
        )
    ax.set_title("选题4：三种书写目标的 ResNet50 视觉特征 t-SNE 分布")
    ax.set_xlabel("t-SNE X")
    ax.set_ylabel("t-SNE Y")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    out = DIM2_DIR / f"{MODEL_TAG}_t-SNE_可视化.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float((2 * u) / (len(x) * len(y)) - 1)


def run_topic4_stats(index_df: pd.DataFrame, pcs: np.ndarray, coords: np.ndarray) -> None:
    groups = index_df["group"].values

    kruskal_rows = []
    for pc_idx in range(min(10, pcs.shape[1])):
        arrs = [pcs[groups == group, pc_idx] for group in GROUP_ORDER]
        h_stat, p_value = stats.kruskal(*arrs)
        kruskal_rows.append(
            {
                "dimension": f"PC{pc_idx + 1}",
                "kruskal_H": h_stat,
                "p_value": p_value,
            }
        )
    pd.DataFrame(kruskal_rows).to_csv(
        STATS_DIR / f"{MODEL_TAG}_kruskal_top10_pcs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_rows = []
    pairs = [("端正易读", "美观"), ("端正易读", "极致扭曲"), ("美观", "极致扭曲")]
    for left, right in pairs:
        left_mask = groups == left
        right_mask = groups == right
        centroid_left = pcs[left_mask, :10].mean(axis=0)
        centroid_right = pcs[right_mask, :10].mean(axis=0)
        pair_rows.append(
            {
                "pair": f"{left} vs {right}",
                "centroid_distance_top10PC": float(np.linalg.norm(centroid_left - centroid_right)),
                "centroid_distance_tSNE": float(
                    np.linalg.norm(coords[left_mask].mean(axis=0) - coords[right_mask].mean(axis=0))
                ),
                "PC1_cliffs_delta": cliffs_delta(pcs[left_mask, 0], pcs[right_mask, 0]),
            }
        )
    pd.DataFrame(pair_rows).to_csv(
        STATS_DIR / f"{MODEL_TAG}_pairwise_group_distances.csv",
        index=False,
        encoding="utf-8-sig",
    )

    x = pcs[:, : min(20, pcs.shape[1])]
    y = groups
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    report = (
        f"Topic 4 classification using {MODEL_TAG} PCA features\n"
        f"Accuracy: {accuracy_score(y_test, pred):.4f}\n"
        f"Macro F1: {f1_score(y_test, pred, average='macro'):.4f}\n\n"
        + classification_report(y_test, pred, labels=GROUP_ORDER)
    )
    (STATS_DIR / f"{MODEL_TAG}_classification_metrics.txt").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    index_df = list_all_images()
    print(index_df.groupby("group").size())
    features = extract_features(index_df)
    print(f"features: {write_feature_csv(index_df, features)}")
    pcs, pca_out = run_pca(index_df, features)
    print(f"pca: {pca_out}")
    coords, tsne_out = run_tsne(index_df, pcs)
    print(f"tsne: {tsne_out}")
    print(f"figure: {plot_tsne(index_df, coords)}")
    run_topic4_stats(index_df, pcs, coords)
    print(f"stats: {STATS_DIR}")


if __name__ == "__main__":
    main()
