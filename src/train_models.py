# -*- coding: utf-8 -*-
"""Train/evaluate Topic 4 and Topic 5 classifiers from prepared features."""

from __future__ import annotations

import argparse
import itertools
import re
import shutil
from pathlib import Path

from utils import (
    CASE_DIR,
    DATA_DIR,
    LABELS,
    TOPIC4_DIR,
    TOPIC5_DIR,
    add_local_pydeps,
    ensure_output_dirs,
    list_data_images,
)
from visualization import save_confusion_matrix, save_image_sheet, save_metric_bars, save_scatter, setup_chinese_font
from morphology_features import MORPHOLOGY_FEATURES, load_existing_morphology

add_local_pydeps()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TOPIC4_STATS = CASE_DIR / "4_选题4统计检验"
RESNET_PCA = CASE_DIR / "2_视觉主成分" / "resnet50_PCA_features_part0001.csv"
RESNET_TSNE = CASE_DIR / "3_降维到二维平面" / "resnet50_tSNE.csv"
SWIN_PCA_BINARY = CASE_DIR / "1_视觉特征" / "topic5_swin_t_PCA_binary.csv"


def page_id_from_filename(filename: str) -> str:
    match = re.search(r"_p(\d+)_([LR])_", filename)
    if match:
        return f"p{int(match.group(1)):02d}_{match.group(2)}"
    return ""


def train_test_indices(df: pd.DataFrame):
    groups = df["filename"].map(page_id_from_filename)
    if groups.nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        return next(splitter.split(df, df["group"], groups))
    return train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["group"],
    )


def cliffs_delta(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / max(len(a) * len(b), 1))


def fit_random_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[str],
    split_indices=None,
):
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["group"].to_numpy()
    train_idx, test_idx = split_indices if split_indices is not None else train_test_indices(df)
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    metrics = {
        "accuracy": accuracy_score(y[test_idx], pred),
        "macro_f1": f1_score(y[test_idx], pred, average="macro"),
        "n_samples": len(df),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
    cm = confusion_matrix(y[test_idx], pred, labels=labels)
    report = classification_report(y[test_idx], pred, labels=labels, zero_division=0)
    pred_df = df.iloc[test_idx][["filename", "group"]].copy()
    pred_df["predicted"] = pred
    pred_df["correct"] = pred_df["group"] == pred_df["predicted"]
    return clf, metrics, cm, report, pred_df


def load_resnet_pca() -> pd.DataFrame:
    if not RESNET_PCA.exists():
        raise FileNotFoundError(f"缺少 ResNet50 PCA 文件，请先运行基础案例分析: {RESNET_PCA}")
    return pd.read_csv(RESNET_PCA)


def make_sample_figure(out: Path, labels=LABELS, n_each: int = 6) -> None:
    rows = []
    for label in labels:
        images = [r for r in list_data_images(DATA_DIR) if r["label"] == label][:n_each]
        for row in images:
            rows.append({"image_path": row["image_path"], "caption": label})
    save_image_sheet(rows, out, "三类书写目标裁切样例", cols=n_each)


def compute_topic4_distances(pca_df: pd.DataFrame, tsne_df: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    pc_cols = [f"PC{i}" for i in range(1, 11)]
    for a, b in itertools.combinations(LABELS, 2):
        pa = pca_df[pca_df["group"] == a]
        pb = pca_df[pca_df["group"] == b]
        pc_dist = np.linalg.norm(pa[pc_cols].mean().values - pb[pc_cols].mean().values)
        tsne_dist = np.nan
        if tsne_df is not None:
            ta = tsne_df[tsne_df["group"] == a]
            tb = tsne_df[tsne_df["group"] == b]
            x_col = "TSNE_X" if "TSNE_X" in tsne_df.columns else "x"
            y_col = "TSNE_Y" if "TSNE_Y" in tsne_df.columns else "y"
            tsne_dist = np.linalg.norm(ta[[x_col, y_col]].mean().values - tb[[x_col, y_col]].mean().values)
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "centroid_distance_top10PC": pc_dist,
                "centroid_distance_tSNE": tsne_dist,
                "PC1_cliffs_delta": cliffs_delta(pa["PC1"], pb["PC1"]),
            }
        )
    return pd.DataFrame(rows)


def run_topic4() -> None:
    ensure_output_dirs()
    fig_dir = TOPIC4_DIR / "figures"
    table_dir = TOPIC4_DIR / "tables"
    metric_dir = TOPIC4_DIR / "metrics"
    pca_df = load_resnet_pca()
    pc_cols = [c for c in pca_df.columns if c.startswith("PC")]

    make_sample_figure(fig_dir / "topic4_sample_examples.png")
    save_scatter(pca_df, "PC1", "PC2", "group", "选题4 ResNet50 PCA 分布", fig_dir / "topic4_resnet50_pca_scatter.png")
    tsne_df = pd.read_csv(RESNET_TSNE) if RESNET_TSNE.exists() else None
    if tsne_df is not None:
        x_col = "TSNE_X" if "TSNE_X" in tsne_df.columns else "x"
        y_col = "TSNE_Y" if "TSNE_Y" in tsne_df.columns else "y"
        save_scatter(tsne_df, x_col, y_col, "group", "选题4 ResNet50 t-SNE 分布", fig_dir / "topic4_resnet50_tsne_scatter.png")
        src = CASE_DIR / "3_降维到二维平面" / "resnet50_t-SNE_可视化.png"
        if src.exists():
            shutil.copy2(src, fig_dir / "topic4_resnet50_tsne_thumbnail_map.png")

    clf, metrics, cm, report, _pred = fit_random_forest(pca_df, pc_cols[:50], LABELS)
    save_confusion_matrix(cm, LABELS, "选题4三分类混淆矩阵", fig_dir / "topic4_confusion_matrix.png")
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(table_dir / "topic4_confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame([{"model": "ResNet50_PCA_RandomForest", **metrics}]).to_csv(
        metric_dir / "topic4_classification_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (metric_dir / "topic4_classification_report.txt").write_text(report, encoding="utf-8")
    distances = compute_topic4_distances(pca_df, tsne_df)
    distances.to_csv(table_dir / "topic4_group_distances.csv", index=False, encoding="utf-8-sig")
    if (TOPIC4_STATS / "resnet50_kruskal_top10_pcs.csv").exists():
        shutil.copy2(TOPIC4_STATS / "resnet50_kruskal_top10_pcs.csv", table_dir / "topic4_kruskal_top10_pcs.csv")


def load_resnet_binary() -> pd.DataFrame:
    df = load_resnet_pca()
    df = df[df["group"].isin(["美观", "极致扭曲"])].copy()
    rename = {f"PC{i}": f"resnet50_PC{i}" for i in range(1, 51) if f"PC{i}" in df.columns}
    return df.rename(columns=rename)


def load_swin_binary() -> pd.DataFrame:
    if not SWIN_PCA_BINARY.exists():
        raise FileNotFoundError(f"缺少 Swin_T PCA 文件，请先运行 topic5 模型对比脚本: {SWIN_PCA_BINARY}")
    df = pd.read_csv(SWIN_PCA_BINARY)
    return df[df["group"].isin(["美观", "极致扭曲"])].copy()


def build_topic5_frames():
    resnet = load_resnet_binary()
    swin = load_swin_binary()
    morph = load_existing_morphology()
    morph_cols = [c for c in MORPHOLOGY_FEATURES if c in morph.columns]
    morph = morph[["filename", "group"] + morph_cols].dropna().copy()
    resnet_cols = [c for c in resnet.columns if c.startswith("resnet50_PC")][:20]
    swin_cols = [c for c in swin.columns if c.startswith("swin_t_PC")][:20]

    # Topic 5 compares all feature schemes on the same valid sample set.
    # The morphology table defines the valid copied cells; deep features are
    # filtered to that exact image id set and ordered identically.
    key_cols = ["filename", "group"]
    base = morph[key_cols].drop_duplicates().copy()
    base = base.merge(resnet[key_cols].drop_duplicates(), on=key_cols, how="inner")
    base = base.merge(swin[key_cols].drop_duplicates(), on=key_cols, how="inner")
    base = base.sort_values(key_cols).reset_index(drop=True)

    resnet_common = base.merge(resnet[key_cols + resnet_cols], on=key_cols, how="left")
    swin_common = base.merge(swin[key_cols + swin_cols], on=key_cols, how="left")
    morph_common = base.merge(morph[key_cols + morph_cols], on=key_cols, how="left")
    fusion = morph_common.merge(resnet_common[key_cols + resnet_cols], on=key_cols, how="left")
    fusion = fusion.merge(swin_common[key_cols + swin_cols], on=key_cols, how="left")

    return {
        "ResNet50": (resnet_common, resnet_cols),
        "Swin_T": (swin_common, swin_cols),
        "Morphology": (morph_common, morph_cols),
        "Fusion": (fusion, morph_cols + resnet_cols + swin_cols),
    }


def plot_feature_spaces(frames, out: Path) -> None:
    setup_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2))
    axes = axes.ravel()
    for ax, (name, (df, cols)) in zip(axes, frames.items()):
        X = df[cols].to_numpy(dtype=float)
        if name in {"Morphology", "Fusion"}:
            X = StandardScaler().fit_transform(X)
            xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
            x_label, y_label = "PCA-1", "PCA-2"
        else:
            xy = X[:, :2]
            x_label, y_label = cols[0], cols[1]
        tmp = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "group": df["group"].values})
        for label, part in tmp.groupby("group"):
            color = "#059669" if label == "美观" else "#dc2626"
            ax.scatter(part["x"], part["y"], s=9, alpha=0.55, label=label, c=color, linewidths=0)
        ax.set_title(name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _feature_space_xy(name: str, df: pd.DataFrame, cols: list[str]):
    X = df[cols].to_numpy(dtype=float)
    if name in {"Morphology", "Fusion"}:
        X = StandardScaler().fit_transform(X)
        xy = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
        return xy, "PCA-1", "PCA-2"
    return X[:, :2], cols[0], cols[1]


def plot_individual_feature_spaces(frames, fig_dir: Path) -> None:
    for name in ("ResNet50", "Swin_T", "Morphology"):
        df, cols = frames[name]
        xy, x_label, y_label = _feature_space_xy(name, df, cols)
        plot_df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "group": df["group"].values})
        save_scatter(
            plot_df,
            "x",
            "y",
            "group",
            f"选题5 {name} 特征二维分布",
            fig_dir / f"topic5_{name.lower()}_feature_space.png",
        )


def plot_confusion_grid(cms: dict[str, np.ndarray], labels: list[str], out: Path) -> None:
    setup_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks(range(len(labels)), labels=labels)
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        ax.tick_params(labelsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_feature_importance(clf, cols: list[str], out_png: Path, out_csv: Path) -> None:
    setup_chinese_font()
    imp = pd.DataFrame({"feature": cols, "importance": clf.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(out_csv, index=False, encoding="utf-8-sig")
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.barh(top["feature"], top["importance"], color="#2563eb")
    ax.set_title("Fusion 模型 Top 15 特征重要性")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def image_path_for_filename(filename: str) -> str:
    rows = list_data_images(DATA_DIR)
    lookup = {r["filename"]: r["image_path"] for r in rows}
    return lookup.get(filename, "")


def run_topic5() -> None:
    ensure_output_dirs()
    fig_dir = TOPIC5_DIR / "figures"
    table_dir = TOPIC5_DIR / "tables"
    metric_dir = TOPIC5_DIR / "metrics"
    labels = ["美观", "极致扭曲"]
    frames = build_topic5_frames()
    make_sample_figure(fig_dir / "topic5_aesthetic_vs_distorted_examples.png", labels=labels, n_each=8)
    common_df = frames["Fusion"][0][["filename", "group"]].copy()
    common_split = train_test_indices(common_df)
    split_df = common_df.copy()
    split_df["page_id"] = split_df["filename"].map(page_id_from_filename)
    split_df["split"] = "train"
    split_df.loc[common_split[1], "split"] = "test"
    split_df.to_csv(table_dir / "topic5_common_sample_split.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    reports = []
    cms: dict[str, np.ndarray] = {}
    preds: dict[str, pd.DataFrame] = {}
    clfs = {}
    for name, (df, cols) in frames.items():
        clf, metrics, cm, report, pred_df = fit_random_forest(df, cols, labels, split_indices=common_split)
        clfs[name] = clf
        cms[name] = cm
        preds[name] = pred_df
        metric_rows.append({"model": name, "n_features": len(cols), "split": "GroupShuffleSplit(page_id, 8:2)", **metrics})
        reports.append(f"===== {name} =====\n{report}\n")
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            table_dir / f"topic5_{name}_confusion_matrix.csv", encoding="utf-8-sig"
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(table_dir / "topic5_model_performance.csv", index=False, encoding="utf-8-sig")
    (metric_dir / "topic5_classification_reports.txt").write_text("\n".join(reports), encoding="utf-8")
    save_metric_bars(metrics_df, fig_dir / "topic5_accuracy_macro_f1_bars.png")
    plot_feature_spaces(frames, fig_dir / "topic5_feature_space_comparison.png")
    plot_individual_feature_spaces(frames, fig_dir)
    plot_confusion_grid(cms, labels, fig_dir / "topic5_confusion_matrices.png")

    fusion_df, fusion_cols = frames["Fusion"]
    plot_feature_importance(
        clfs["Fusion"],
        fusion_cols,
        fig_dir / "topic5_fusion_top15_feature_importance.png",
        table_dir / "topic5_fusion_feature_importance.csv",
    )

    mis = preds["Fusion"][~preds["Fusion"]["correct"]].copy()
    mis["page_id"] = mis["filename"].map(page_id_from_filename)
    mis["image_path"] = mis["filename"].map(image_path_for_filename)
    mis.to_csv(table_dir / "topic5_misclassified_samples.csv", index=False, encoding="utf-8-sig")
    examples = []
    for true_label, pred_label in [("美观", "极致扭曲"), ("极致扭曲", "美观")]:
        part = mis[(mis["group"] == true_label) & (mis["predicted"] == pred_label)].head(8)
        for _, row in part.iterrows():
            examples.append(
                {
                    "image_path": row["image_path"],
                    "caption": f"{row['group']} → {row['predicted']}",
                }
            )
    save_image_sheet(examples, fig_dir / "topic5_misclassified_examples.png", "Fusion 典型误判样本", cols=8)

    # Also copy earlier generated comparison plots if present for traceability.
    old_dir = CASE_DIR / "4_选题4统计检验"
    for old, new in {
        "topic5_model_accuracy_comparison.png": "topic5_previous_model_accuracy_comparison.png",
        "topic5_fusion_feature_importance.png": "topic5_previous_fusion_feature_importance.png",
    }.items():
        src = old_dir / old
        if src.exists():
            shutil.copy2(src, fig_dir / new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["topic4", "topic5", "all"], default="all")
    args = parser.parse_args()
    if args.task in {"topic4", "all"}:
        run_topic4()
    if args.task in {"topic5", "all"}:
        run_topic5()
    print("analysis outputs updated under outputs/")


if __name__ == "__main__":
    main()
