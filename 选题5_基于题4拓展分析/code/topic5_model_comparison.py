# -*- coding: utf-8 -*-
"""Topic 5 extension: visual-model comparison for aesthetic vs distorted writing.

This experiment extends Topic 4. Topic 4 established that the three copying
goals differ in visual feature space. Topic 5 focuses on the hardest pair:
"美观" vs "极致扭曲". It compares:

1. ResNet50 PCA features produced by the Topic 4 visual pipeline.
2. Swin_T features extracted from the local torchvision checkpoint.
3. Interpretable morphology features from the earlier deformation analysis.
4. A fusion model combining Swin/ResNet deep features and morphology features.
"""

from __future__ import annotations

import csv
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TOPIC5_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = TOPIC5_ROOT.parent
TOPIC4_ROOT = WORKSPACE / "基础案例A-视觉分析2"
MODEL_ROOT = WORKSPACE / "视觉特征计算" / "model"
RESULTS_DIR = TOPIC4_ROOT / "4_选题4统计检验"
PROCESSED_DIR = TOPIC4_ROOT / "1_视觉特征"
MORPH_DIR = TOPIC5_ROOT / "data" / "processed"

for directory in (RESULTS_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

GROUPS = ["美观", "极致扭曲"]
GROUP_COLORS = {"美观": "#10B981", "极致扭曲": "#EF4444"}
GROUP_TO_BINARY = {"美观": 0, "极致扭曲": 1}

BATCH_SIZE = 64
PCA_COMPONENTS = 50
CLASSIFIER_PCS = 20

MORPH_FEATURES = [
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
    "ref_iou",
    "ref_cosine",
    "ref_chamfer",
]


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_topic4_index() -> pd.DataFrame:
    df = pd.read_csv(TOPIC4_ROOT / "1_视觉特征" / "image_index.csv")
    df = df[df["group"].isin(GROUPS)].copy()
    df["abs_path"] = df["path"].apply(lambda p: str(TOPIC4_ROOT / p))
    return df.reset_index(drop=True)


def find_swin_weight() -> Path:
    candidates = [
        MODEL_ROOT / "torchvision" / "Swin_T.pth",
        MODEL_ROOT / "torch" / "hub" / "checkpoints" / "swin_t-704ceda3.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Swin_T checkpoint not found")


def extract_swin_features(index_df: pd.DataFrame) -> pd.DataFrame:
    out_path = PROCESSED_DIR / "topic5_swin_t_features_binary.csv"
    if out_path.exists() and out_path.stat().st_size > 100:
        print(f"Using cached Swin_T features: {out_path}")
        return pd.read_csv(out_path)

    weight_path = find_swin_weight()
    model = models.swin_t(weights=None)
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.head = torch.nn.Identity()
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

    paths = [Path(p) for p in index_df["abs_path"].tolist()]
    features = []
    total = len(paths)
    print(f"Extracting Swin_T features for {total} binary samples on {device}...")
    for start in range(0, total, BATCH_SIZE):
        batch_paths = paths[start : start + BATCH_SIZE]
        batch = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        ).to(device)
        with torch.no_grad():
            out = model(batch).cpu().numpy().astype(np.float32)
        features.append(out)
        done = min(start + BATCH_SIZE, total)
        if done % 640 == 0 or done == total:
            print(f"  {done}/{total}")

    arr = np.vstack(features)
    df = pd.DataFrame(arr, columns=[f"swin_f{i + 1}" for i in range(arr.shape[1])])
    df.insert(0, "group", index_df["group"].values)
    df.insert(0, "filename", index_df["filename"].values)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def pca_features(df: pd.DataFrame, prefix: str, feature_prefix: str) -> pd.DataFrame:
    out_path = PROCESSED_DIR / f"topic5_{prefix}_PCA_binary.csv"
    var_path = RESULTS_DIR / f"{prefix}_PCA_variance.csv"
    if out_path.exists() and out_path.stat().st_size > 100:
        return pd.read_csv(out_path)

    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    x = df[feature_cols].to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=min(PCA_COMPONENTS, x.shape[1], x.shape[0] - 1), random_state=SEED)
    pcs = pca.fit_transform(x)
    out = pd.DataFrame(pcs, columns=[f"{prefix}_PC{i + 1}" for i in range(pcs.shape[1])])
    out.insert(0, "group", df["group"].values)
    out.insert(0, "filename", df["filename"].values)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(var_path, index=False, encoding="utf-8-sig")
    return out


def load_resnet_pca() -> pd.DataFrame:
    df = pd.read_csv(TOPIC4_ROOT / "2_视觉主成分" / "resnet50_PCA_features_part0001.csv")
    df = df[df["group"].isin(GROUPS)].copy()
    rename = {col: f"resnet50_{col}" for col in df.columns if col.startswith("PC")}
    return df.rename(columns=rename).reset_index(drop=True)


def load_morphology_features() -> pd.DataFrame:
    path = MORPH_DIR / "handwriting_deformation_features_valid.csv"
    df = pd.read_csv(path)
    df = df[df["style_label"].isin(GROUPS)].copy()
    df["filename"] = df.apply(
        lambda row: (
            f"{row['style_label']}_p{int(row['page']):02d}_{row['panel']}"
            f"_r{int(row['row_in_panel']):02d}_rep{int(row['replicate'])}"
        ),
        axis=1,
    )
    df["group"] = df["style_label"]
    cols = ["filename", "group"] + MORPH_FEATURES
    return df[cols].reset_index(drop=True)


def classify_dataset(name: str, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    x = df[feature_cols].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["group"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    labels = GROUPS
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(y_test, pred, labels=labels)
    metrics = {
        "model": name,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importance": list(zip(feature_cols, clf.feature_importances_)),
    }
    return metrics


def separability_stats(name: str, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    x = df[feature_cols].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    labels = df["group"].to_numpy()
    left = x[labels == "美观"]
    right = x[labels == "极致扭曲"]
    centroid_distance = float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))
    # Test first available dimension as a compact scalar reference.
    u, p_value = stats.mannwhitneyu(left[:, 0], right[:, 0], alternative="two-sided")
    cliffs = float((2 * u) / (len(left) * len(right)) - 1)
    return {
        "model": name,
        "centroid_distance": centroid_distance,
        "first_dim_mannwhitney_p": p_value,
        "first_dim_cliffs_delta": cliffs,
    }


def write_metrics(metrics_list: list[dict], stats_list: list[dict]) -> None:
    summary_path = RESULTS_DIR / "topic5_model_comparison_metrics.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "n_samples", "n_features", "accuracy", "macro_f1"],
        )
        writer.writeheader()
        for item in metrics_list:
            writer.writerow(
                {
                    "model": item["model"],
                    "n_samples": item["n_samples"],
                    "n_features": item["n_features"],
                    "accuracy": f"{item['accuracy']:.6f}",
                    "macro_f1": f"{item['macro_f1']:.6f}",
                }
            )

    stats_path = RESULTS_DIR / "topic5_separability_stats.csv"
    pd.DataFrame(stats_list).to_csv(stats_path, index=False, encoding="utf-8-sig")

    text_path = RESULTS_DIR / "topic5_classification_reports.txt"
    with text_path.open("w", encoding="utf-8") as f:
        for item in metrics_list:
            f.write(f"=== {item['model']} ===\n")
            f.write(f"Accuracy: {item['accuracy']:.4f}\n")
            f.write(f"Macro F1: {item['macro_f1']:.4f}\n")
            f.write(item["classification_report"])
            f.write("\n\n")

    for item in metrics_list:
        imp_path = RESULTS_DIR / f"{item['model']}_feature_importance.csv"
        imps = sorted(item["feature_importance"], key=lambda pair: pair[1], reverse=True)
        pd.DataFrame(imps, columns=["feature", "importance"]).to_csv(
            imp_path,
            index=False,
            encoding="utf-8-sig",
        )


def plot_model_comparison(metrics_list: list[dict]) -> Path:
    set_chinese_font()
    names = [item["model"] for item in metrics_list]
    acc = [item["accuracy"] for item in metrics_list]
    f1 = [item["macro_f1"] for item in metrics_list]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, acc, width, label="Accuracy", color="#2563EB", alpha=0.86)
    ax.bar(x + width / 2, f1, width, label="Macro F1", color="#F97316", alpha=0.86)
    ax.set_ylim(0, 1)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_title("选题5：美观 vs 极致扭曲的模型识别效果")
    ax.set_ylabel("score")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(acc):
        ax.text(i - width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    for i, value in enumerate(f1):
        ax.text(i + width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / "topic5_model_accuracy_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_confusion_matrices(metrics_list: list[dict]) -> Path:
    set_chinese_font()
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(4.0 * len(metrics_list), 3.8))
    if len(metrics_list) == 1:
        axes = [axes]
    for ax, item in zip(axes, metrics_list):
        cm = item["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(item["model"])
        ax.set_xticks([0, 1], GROUPS, rotation=25, ha="right")
        ax.set_yticks([0, 1], GROUPS)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.suptitle("美观 vs 极致扭曲：混淆矩阵", fontsize=14)
    fig.tight_layout()
    out = RESULTS_DIR / "topic5_confusion_matrices.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_feature_space(resnet_df: pd.DataFrame, swin_pca: pd.DataFrame, morph: pd.DataFrame) -> Path:
    set_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [
        ("ResNet50 PCA", resnet_df, "resnet50_PC1", "resnet50_PC2"),
        ("Swin_T PCA", swin_pca, "swin_t_PC1", "swin_t_PC2"),
        ("形态特征 PCA", pca_morph_for_plot(morph), "morph_PC1", "morph_PC2"),
    ]
    for ax, (title, df, x_col, y_col) in zip(axes, panels):
        for group in GROUPS:
            sub = df[df["group"] == group]
            if len(sub) > 1500:
                sub = sub.sample(1500, random_state=SEED)
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=8,
                alpha=0.42,
                c=GROUP_COLORS[group],
                label=group,
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("不同特征空间中的美观/极致扭曲分布", fontsize=14)
    fig.tight_layout()
    out = RESULTS_DIR / "topic5_feature_space_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def pca_morph_for_plot(morph: pd.DataFrame) -> pd.DataFrame:
    x = StandardScaler().fit_transform(morph[MORPH_FEATURES].to_numpy(dtype=float))
    pcs = PCA(n_components=2, random_state=SEED).fit_transform(x)
    out = pd.DataFrame({"filename": morph["filename"], "group": morph["group"], "morph_PC1": pcs[:, 0], "morph_PC2": pcs[:, 1]})
    return out


def plot_fusion_importance(fusion_metrics: dict) -> Path:
    set_chinese_font()
    imps = sorted(fusion_metrics["feature_importance"], key=lambda pair: pair[1], reverse=True)[:14]
    labels = [name for name, _ in imps][::-1]
    values = [value for _, value in imps][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.barh(labels, values, color="#0F766E", alpha=0.86)
    ax.set_title("融合模型的重要特征 Top 14")
    ax.set_xlabel("Random Forest importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = RESULTS_DIR / "topic5_fusion_feature_importance.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    index_df = read_topic4_index()
    print(index_df.groupby("group").size())

    resnet_pca = load_resnet_pca()
    resnet_cols = [f"resnet50_PC{i}" for i in range(1, CLASSIFIER_PCS + 1)]

    swin_features = extract_swin_features(index_df)
    swin_pca = pca_features(swin_features, "swin_t", "swin_f")
    swin_cols = [f"swin_t_PC{i}" for i in range(1, CLASSIFIER_PCS + 1)]

    morph = load_morphology_features()

    # Inner-join for fusion: only valid morphology samples are kept.
    fusion = (
        morph.merge(resnet_pca[["filename", "group"] + resnet_cols], on=["filename", "group"], how="inner")
        .merge(swin_pca[["filename", "group"] + swin_cols], on=["filename", "group"], how="inner")
    )
    fusion_path = PROCESSED_DIR / "topic5_fusion_features_binary.csv"
    fusion.to_csv(fusion_path, index=False, encoding="utf-8-sig")

    metrics = [
        classify_dataset("ResNet50", resnet_pca, resnet_cols),
        classify_dataset("Swin_T", swin_pca, swin_cols),
        classify_dataset("Morphology", morph, MORPH_FEATURES),
        classify_dataset("Fusion", fusion, MORPH_FEATURES + resnet_cols + swin_cols),
    ]
    sep_stats = [
        separability_stats("ResNet50", resnet_pca, resnet_cols[:10]),
        separability_stats("Swin_T", swin_pca, swin_cols[:10]),
        separability_stats("Morphology", morph, MORPH_FEATURES),
        separability_stats("Fusion", fusion, MORPH_FEATURES + resnet_cols[:10] + swin_cols[:10]),
    ]

    write_metrics(metrics, sep_stats)
    plot_model_comparison(metrics)
    plot_confusion_matrices(metrics)
    plot_feature_space(resnet_pca, swin_pca, morph)
    plot_fusion_importance(metrics[-1])
    print(f"fusion features: {fusion_path}")
    print(f"results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
