# -*- coding: utf-8 -*-
"""
===========================================
💡 一体化视觉特征分析流水线（分块版）
功能包括：
1️⃣ ViT/CLIP 特征提取（每10000张图片写一个CSV）
2️⃣ PCA 主成分计算（IncrementalPCA，输出与特征分块一一对应）
3️⃣ t-SNE / UMAP 降维（基于全部PCA转化后的数据）
4️⃣ 降维结果图片可视化（缩略图拼贴）
===========================================

更新要点：
- 新增 CHUNK_SIZE=10000 的分块写入逻辑，特征与PCA结果严格对齐（part0001、part0002、...）
- 引入 IncrementalPCA，两遍扫描特征CSV：第一遍 partial_fit，第二遍 transform 并分块写出
- 进度显示：四阶段、模型级、图片级、分块保存、CSV聚合/读取、降维进度
- 严格离线：TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=1 + local_files_only=True + cache_dir
"""

import os
import sys
import math
import json
import logging
import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

# ------------------------------
# 离线环境（必须已预下载模型）
# ------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ------------------------------
# 日志与进度条
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
try:
    from tqdm import tqdm
except Exception:
    raise RuntimeError("缺少依赖：tqdm。请先安装：pip install tqdm")

# ------------------------------------------------------------
# 🔧 全局配置
# ------------------------------------------------------------

# ==== 路径 ====
DATA_ROOT = "Data"                       # 图片主目录（包含多个子文件夹）
FEATURE_DIR = "1_视觉特征"               # 特征输出目录（分块）
PCA_DIR = "2_视觉主成分"                 # PCA结果与图表输出目录（分块）
DIM2_DIR = "3_降维到二维平面"            # t-SNE / UMAP结果与可视化图
CACHE_DIR = "./models/huggingface"       # HF 模型本地缓存目录
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(PCA_DIR, exist_ok=True)
os.makedirs(DIM2_DIR, exist_ok=True)

# ==== 分块 ====
CHUNK_SIZE = 10000                       # 每10000幅图写一个CSV
INDEX_FILE = os.path.join(FEATURE_DIR, "image_index.csv")  # 全局索引（用于对齐）

# ==== 模型控制 ====
USE_VIT = True
USE_CLIP = True
# MODELS 会与开关同步生成，避免“有名无实”的混淆
MODELS = []
if USE_VIT:  MODELS.append("vit")
if USE_CLIP: MODELS.append("clip")

# ==== 算法开关 ====
RUN_FEATURE = True       # 提取视觉特征（分块输出）
RUN_PCA = True           # 增量PCA（两遍，分块输出）+ 解释率表
RUN_TSNE_UMAP = True     # t-SNE / UMAP 降维（基于全部 PCA 的拼接）
RUN_VISUALIZE = True     # 二维平面可视化（拼贴缩略图）

# ==== 降维参数 ====
USE_TSNE = True
USE_UMAP = True
TSNE_PERPLEXITY = 15
TSNE_ITER = 1000
TSNE_SEED = 42

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_SEED = 42

# ==== 可视化参数 ====
CANVAS_SIZE = (2000, 2000)
THUMB_SIZE = (40, 40)
MARGIN = 150
BG_COLOR = (255, 255, 255)
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"  # Windows 默认；找不到会fallback
BORDER_THICKNESS = 2
BORDER_SAT = 0.9
BORDER_VAL = 0.9

# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def ensure_cached(model_names, what="model/processor"):
    """确保模型/处理器已缓存到 CACHE_DIR，否则给出清晰报错。"""
    missing = []
    for name in model_names:
        # 不是严格校验：提示用户“需已缓存”，否则 HF 会在本地查找失败
        # 这里做弱校验：缓存目录是否存在
        if not Path(CACHE_DIR).exists():
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"未找到本地缓存目录 {CACHE_DIR}。请先在联网环境下将以下{what}预下载至该目录：\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n并确保本程序离线读取时可用。"
        )

def list_all_images(data_root):
    """扫描 Data 下的所有图片，返回 [(path, filename_wo_ext, group)]。"""
    all_images = []
    groups = []
    if not Path(data_root).exists():
        raise RuntimeError(f"数据根目录不存在：{data_root}")
    for group_name in sorted(os.listdir(data_root)):
        gpath = os.path.join(data_root, group_name)
        if not os.path.isdir(gpath):
            continue
        for fn in sorted(os.listdir(gpath)):
            low = fn.lower()
            if low.endswith((".jpg", ".jpeg", ".png")):
                all_images.append((os.path.join(gpath, fn), os.path.splitext(fn)[0], group_name))
                groups.append(group_name)
    return all_images, sorted(set(groups))

def save_index_csv(all_images):
    """保存全局索引，便于复现和对齐。"""
    df = pd.DataFrame(all_images, columns=["path", "filename", "group"])
    df.to_csv(INDEX_FILE, index=False, encoding="utf-8-sig")
    return df

def chunk_writer(base_dir, model, part_idx, names, groups, feats):
    """将一个分块写入 CSV。"""
    out = os.path.join(base_dir, f"{model}_features_part{part_idx:04d}.csv")
    df = pd.DataFrame(feats)
    df.insert(0, "group", groups)
    df.insert(0, "filename", names)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logging.info(f"  💾 {model} 特征分块已写出：{out}（{len(df)} 行）")

def chunk_pca_writer(base_dir, model, part_idx, names, groups, pcs):
    """将一个PCA分块写入 CSV。"""
    df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
    df.insert(0, "group", groups)
    df.insert(0, "filename", names)
    out = os.path.join(base_dir, f"{model}_PCA_features_part{part_idx:04d}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logging.info(f"  💾 {model} PCA分块已写出：{out}（{len(df)} 行）")

def generate_colors(groups):
    uniq = sorted(set(groups))
    cmap = {}
    for i, g in enumerate(uniq):
        hue = i / max(1, len(uniq))
        r, g_, b = colorsys.hsv_to_rgb(hue, BORDER_SAT, BORDER_VAL)
        cmap[g] = (int(255*r), int(255*g_), int(255*b))
    return cmap

def read_all_pca_parts(model):
    """读取某模型的所有 PCA 分块并拼接，返回 DataFrame（filename, group, PCs...）"""
    parts = sorted(Path(PCA_DIR).glob(f"{model}_PCA_features_part*.csv"))
    if not parts:
        logging.warning(f"⚠️ 未找到 {model} 的 PCA 分块文件。")
        return None
    dfs = []
    for p in tqdm(parts, desc=f"读取 {model} PCA 分块", unit="part"):
        df = pd.read_csv(p)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)

# ------------------------------------------------------------
# 1️⃣ 特征提取（ViT / CLIP）—— 分块写出
# ------------------------------------------------------------
if RUN_FEATURE:
    logging.info("===== 阶段1：特征提取（分块） =====")
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🧠 当前计算设备: {device}")

    # 扫描图片与组
    all_images, uniq_groups = list_all_images(DATA_ROOT)
    total_images = len(all_images)
    logging.info(f"📸 检测到 {total_images} 张图片，分属 {len(uniq_groups)} 个子文件夹（组）。")

    if total_images == 0:
        raise RuntimeError("未在 Data/ 下找到任何 jpg/png 图片。")

    # 保存全局索引（可复用）
    index_df = save_index_csv(all_images)
    logging.info(f"🧾 全局索引已保存：{INDEX_FILE}")

    # 模型准备（离线缓存检查）
    need_models = []
    if USE_VIT:
        need_models.extend(["google/vit-base-patch16-224-in21k"])
    if USE_CLIP:
        need_models.extend(["openai/clip-vit-base-patch32"])
    ensure_cached(need_models, what="模型/处理器")

    # 导入 HF
    try:
        from transformers import ViTModel, ViTFeatureExtractor, CLIPProcessor, CLIPModel
    except Exception as e:
        raise RuntimeError("请先安装 transformers 与 torch：pip install transformers torch") from e

    # 装载可用的模型
    models_dict = {}
    if USE_VIT:
        vit_name = "google/vit-base-patch16-224-in21k"
        vit_extractor = ViTFeatureExtractor.from_pretrained(
            vit_name, cache_dir=CACHE_DIR, local_files_only=True
        )
        vit_model = ViTModel.from_pretrained(
            vit_name, cache_dir=CACHE_DIR, local_files_only=True
        ).to(device).eval()
        models_dict["vit"] = ("vit", vit_model, vit_extractor)

    if USE_CLIP:
        clip_name = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(
            clip_name, cache_dir=CACHE_DIR, local_files_only=True
        ).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(
            clip_name, cache_dir=CACHE_DIR, local_files_only=True
        )
        models_dict["clip"] = ("clip", clip_model, clip_processor)

    # 定义提取函数
    def extract_features_vit(img_path, model, extractor):
        img = Image.open(img_path).convert("RGB")
        inputs = extractor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return feats

    def extract_features_clip(img_path, model, processor):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            feats = outputs.cpu().numpy().flatten()
        return feats

    # 四阶段中的子进度：模型级 + 图片级
    for key in MODELS:
        if key not in models_dict:
            logging.warning(f"⚠️ MODELS 包含 {key}，但该模型未启用或加载失败。跳过。")
            continue

        logging.info(f"\n🚀 提取 {key.upper()} 特征中 ...")
        model_key, model_obj, model_aux = models_dict[key]

        # 为了避免一次装太多内存，按 CHUNK_SIZE 组批写出
        part_idx = 1
        buf_feats, buf_names, buf_groups = [], [], []

        pbar = tqdm(all_images, desc=f"{key.upper()} 特征提取", unit="img")
        for (img_path, fname, gname) in pbar:
            try:
                if key == "vit":
                    fvec = extract_features_vit(img_path, model_obj, model_aux)
                else:
                    fvec = extract_features_clip(img_path, model_obj, model_aux)
            except Exception as e:
                logging.warning(f"  ⚠️ 无法处理 {img_path}: {e}")
                continue

            buf_feats.append(fvec)
            buf_names.append(fname)
            buf_groups.append(gname)

            # 满一块或最后一批则写出
            if len(buf_feats) >= CHUNK_SIZE:
                chunk_writer(FEATURE_DIR, key, part_idx, buf_names, buf_groups, buf_feats)
                part_idx += 1
                buf_feats, buf_names, buf_groups = [], [], []

        # 尾块写出
        if buf_feats:
            chunk_writer(FEATURE_DIR, key, part_idx, buf_names, buf_groups, buf_feats)

    logging.info("✅ 阶段1完成：特征提取分块已全部写出。")

# ------------------------------------------------------------
# 2️⃣ PCA主成分（自动选择：小样本=标准PCA；大样本=IncrementalPCA）
#     - 与特征分块严格一一对应写出
#     - 生成方差解释率表
# 可调参数：
#   PCA_PLAIN_THRESHOLD：样本量 ≤ 该阈值时使用标准 PCA（一次性 fit）
#   PCA_COMPONENTS_CAP：可选的PC上限（例如 256）；None 表示不额外限缩
# ------------------------------------------------------------
if RUN_PCA:
    logging.info("\n===== 阶段2：PCA（自动选择：标准/增量） =====")

    PCA_PLAIN_THRESHOLD = 20000   # ✅ 样本数 ≤ 2万：用标准PCA；否则用IncrementalPCA
    PCA_COMPONENTS_CAP = None     # 例如设为 256 可提速/稳态；None 表示不过度限缩

    for key in MODELS:
        feat_parts = sorted(Path(FEATURE_DIR).glob(f"{key}_features_part*.csv"))
        if not feat_parts:
            logging.warning(f"⚠️ 未找到 {key} 的特征分块文件，跳过 PCA。")
            continue

        logging.info(f"📦 模型 {key.upper()}：共检测到 {len(feat_parts)} 个特征分块。")

        # 读取首块确定特征维度
        head_df = pd.read_csv(feat_parts[0], nrows=1)
        feat_cols = [c for c in head_df.columns if c not in ("filename", "group")]
        feat_dim = len(feat_cols)

        # 统计总行数
        total_rows = 0
        part_sizes = []
        for p in feat_parts:
            # 更稳妥：用 pandas 读 shape 而不是粗略数行（避免编码/换行差异）
            try:
                df_shape = pd.read_csv(p, usecols=[feat_cols[0]]).shape[0]
            except Exception:
                df_shape = 0
            part_sizes.append(df_shape)
            total_rows += df_shape

        logging.info(f"🧮 {key} 特征总样本数：{total_rows}，特征维度：{feat_dim}")

        if total_rows == 0:
            logging.warning(f"⚠️ {key} 总样本数为 0，跳过 PCA。")
            continue

        # 目标主成分数（不超过样本数与特征维度；还可受 CAP 限制）
        target_components = min(feat_dim, total_rows)
        if PCA_COMPONENTS_CAP is not None:
            target_components = min(target_components, PCA_COMPONENTS_CAP)

        logging.info(
            f"🎯 {key} 目标主成分数 n_components = {target_components} "
            f"（策略：{'标准PCA' if total_rows <= PCA_PLAIN_THRESHOLD else 'IncrementalPCA'}）"
        )

        # ========== 情况A：小样本 —— 标准 PCA ==========
        if total_rows <= PCA_PLAIN_THRESHOLD:
            from sklearn.decomposition import PCA

            # 一次性读入全部特征做 PCA（内存可控）
            logging.info("🔁 标准 PCA：加载全部特征以拟合")
            X_all_list, meta_all = [], []
            for p in tqdm(feat_parts, desc=f"{key.upper()} 读取全部特征以PCA", unit="part"):
                df = pd.read_csv(p)
                if df.shape[0] == 0:
                    continue
                X_all_list.append(df[feat_cols].values)
                meta_all.append(df[["filename", "group"]])

            if len(X_all_list) == 0:
                logging.warning(f"⚠️ {key} 读不到有效特征，跳过。")
                continue

            X_all = np.vstack(X_all_list)
            meta_all_df = pd.concat(meta_all, axis=0, ignore_index=True)

            # 选更稳/快的求解器：当 n_components < min(n_samples, n_features) 且 n_samples 较大时用 randomized
            use_randomized = target_components < min(X_all.shape[0], X_all.shape[1]) and X_all.shape[0] > 1000
            pca = PCA(n_components=target_components, svd_solver=("randomized" if use_randomized else "auto"))
            pca.fit(X_all)

            # 保存方差解释率
            var_ratio = pca.explained_variance_ratio_
            var_df = pd.DataFrame({
                "PC": [f"PC{i+1}" for i in range(len(var_ratio))],
                "explained_variance_ratio": var_ratio
            })
            var_df["cumulative"] = var_df["explained_variance_ratio"].cumsum()
            var_out = os.path.join(PCA_DIR, f"{key}_PCA_variance.csv")
            var_df.to_csv(var_out, index=False, encoding="utf-8-sig")
            logging.info(f"📈 {key} PCA 方差解释率已保存：{var_out}")

            # 仍按分块 transform 并逐块写出（与特征分块一一对应）
            logging.info("🔁 标准 PCA：按分块 transform + 写出")
            part_idx = 1
            for p in tqdm(feat_parts, desc=f"{key.upper()} PCA transform（标准）", unit="part"):
                df = pd.read_csv(p)
                if df.shape[0] == 0:
                    continue
                X = df[feat_cols].values
                pcs = pca.transform(X)
                names = df["filename"].tolist()
                groups = df["group"].tolist()
                chunk_pca_writer(PCA_DIR, key, part_idx, names, groups, pcs)
                part_idx += 1

        # ========== 情况B：大样本 —— IncrementalPCA ==========
        else:
            from sklearn.decomposition import IncrementalPCA

            ipca = IncrementalPCA(n_components=target_components, batch_size=2048)

            # 第一遍：累积分块直到样本数 ≥ n_components，再 partial_fit
            logging.info("🔁 增量 PCA：第1遍 partial_fit（按需累积）")
            buf_X = None
            acc_rows = 0

            for p in tqdm(feat_parts, desc=f"{key.upper()} PCA partial_fit", unit="part"):
                df = pd.read_csv(p)
                if df.shape[0] == 0:
                    continue
                Xp = df[feat_cols].values
                if buf_X is None:
                    buf_X = Xp
                else:
                    buf_X = np.vstack([buf_X, Xp])
                acc_rows = buf_X.shape[0]

                if acc_rows >= target_components:
                    ipca.partial_fit(buf_X)
                    buf_X = None
                    acc_rows = 0

            # 可能有残留不足 n_components 的缓冲；若此前已有一次有效 partial_fit，可再次 partial_fit
            if buf_X is not None and buf_X.shape[0] > 0:
                try:
                    ipca.partial_fit(buf_X)
                except ValueError:
                    logging.warning(
                        f"  ⚠️ 残留样本不足以 partial_fit：{buf_X.shape[0]} < {target_components}，尝试降维兜底。"
                    )
                    new_components = min(target_components, buf_X.shape[0])
                    ipca = IncrementalPCA(n_components=new_components, batch_size=2048)
                    ipca.partial_fit(buf_X)
                    target_components = new_components
                    logging.info(f"  ✅ 已将 n_components 降为 {target_components} 并完成 partial_fit。")

            # 保存方差解释率
            var_ratio = ipca.explained_variance_ratio_
            var_df = pd.DataFrame({
                "PC": [f"PC{i+1}" for i in range(len(var_ratio))],
                "explained_variance_ratio": var_ratio
            })
            var_df["cumulative"] = var_df["explained_variance_ratio"].cumsum()
            var_out = os.path.join(PCA_DIR, f"{key}_PCA_variance.csv")
            var_df.to_csv(var_out, index=False, encoding="utf-8-sig")
            logging.info(f"📈 {key} PCA 方差解释率已保存：{var_out}")

            # 第二遍：transform + 分块写出
            logging.info("🔁 增量 PCA：第2遍 transform + 分块写出")
            part_idx = 1
            for p in tqdm(feat_parts, desc=f"{key.upper()} PCA transform（增量）", unit="part"):
                df = pd.read_csv(p)
                if df.shape[0] == 0:
                    continue
                X = df[feat_cols].values
                pcs = ipca.transform(X)
                names = df["filename"].tolist()
                groups = df["group"].tolist()
                chunk_pca_writer(PCA_DIR, key, part_idx, names, groups, pcs)
                part_idx += 1

    logging.info("✅ 阶段2完成：PCA 分块与方差解释率表已写出。")

# ------------------------------------------------------------
# 3️⃣ 二维降维（t-SNE / UMAP）
#     —— 基于“全部” PCA 分块的合并（内存较大时请考虑采样/分批可视化）
# ------------------------------------------------------------
if RUN_TSNE_UMAP:
    logging.info("\n===== 阶段3：二维降维（t-SNE / UMAP） =====")
    try:
        import umap
        umap_available = True
    except ImportError:
        logging.warning("⚠️ UMAP 模块未安装，将仅执行 t-SNE。")
        umap_available = False
        USE_UMAP = False

    for key in MODELS:
        df_pca_all = read_all_pca_parts(key)
        if df_pca_all is None or df_pca_all.shape[0] == 0:
            continue

        meta_cols = ["filename", "group"]
        pc_cols = [c for c in df_pca_all.columns if c.startswith("PC")]
        X = df_pca_all[pc_cols].values

        # t-SNE
        if USE_TSNE:
            logging.info(f"🌀 运行 t-SNE（{key.upper()}）")
            tsne = TSNE(
                n_components=2,
                random_state=TSNE_SEED,
                perplexity=TSNE_PERPLEXITY,
                n_iter=TSNE_ITER,
                init="pca",
                verbose=1
            )
            emb = tsne.fit_transform(X)
            df_tsne = pd.DataFrame({
                "filename": df_pca_all["filename"],
                "group": df_pca_all["group"],
                "TSNE_X": emb[:, 0],
                "TSNE_Y": emb[:, 1],
            })
            out_tsne = os.path.join(DIM2_DIR, f"{key}_tSNE.csv")
            df_tsne.to_csv(out_tsne, index=False, encoding="utf-8-sig")
            logging.info(f"💾 t-SNE 结果已保存：{out_tsne}")

        # UMAP
        if USE_UMAP and umap_available:
            logging.info(f"🧭 运行 UMAP（{key.upper()}）")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                random_state=UMAP_SEED,
                verbose=True
            )
            emb = reducer.fit_transform(X)
            df_umap = pd.DataFrame({
                "filename": df_pca_all["filename"],
                "group": df_pca_all["group"],
                "UMAP_X": emb[:, 0],
                "UMAP_Y": emb[:, 1],
            })
            out_umap = os.path.join(DIM2_DIR, f"{key}_UMAP.csv")
            df_umap.to_csv(out_umap, index=False, encoding="utf-8-sig")
            logging.info(f"💾 UMAP 结果已保存：{out_umap}")

    logging.info("✅ 阶段3完成：二维降维结果已写出。")

# ------------------------------------------------------------
# 4️⃣ 二维平面可视化（t-SNE / UMAP）
# ------------------------------------------------------------
if RUN_VISUALIZE:
    logging.info("\n===== 阶段4：二维可视化（缩略图拼贴） =====")

    def visualize(df, xcol, ycol, model, method):
        x, y = df[xcol].values, df[ycol].values
        names, groups = df["filename"].tolist(), df["group"].tolist()

        x_min, x_max, y_min, y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        span = max(x_max - x_min, y_max - y_min)
        if span == 0:
            span = 1.0
        x = (x - (x_min + x_max) / 2) / span + 0.5
        y = (y - (y_min + y_max) / 2) / span + 0.5

        cmap = generate_colors(groups)
        canvas = Image.new("RGB", CANVAS_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(canvas)
        try:
            fnt = ImageFont.truetype(FONT_PATH, 42)
        except:
            fnt = ImageFont.load_default()
        title = f"{model.upper()} {method}"
        draw.text((CANVAS_SIZE[0]//2 - min(300, len(title)*10), 40), title, fill=(0,0,0), font=fnt)

        missing = 0
        for xi, yi, n, g in tqdm(list(zip(x, y, names, groups)), desc=f"{model.upper()} {method} 贴图", unit="img"):
            cx = int(MARGIN + xi*(CANVAS_SIZE[0]-2*MARGIN))
            cy = int(CANVAS_SIZE[1]-MARGIN - yi*(CANVAS_SIZE[1]-2*MARGIN))
            jpg = os.path.join(DATA_ROOT, g, f"{n}.jpg")
            png = os.path.join(DATA_ROOT, g, f"{n}.png")
            p = jpg if os.path.exists(jpg) else (png if os.path.exists(png) else None)
            if not p:
                # 可选择：画小点占位
                missing += 1
                r = 3
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=cmap[g], outline=None)
                continue
            try:
                im = Image.open(p).convert("RGB").resize(THUMB_SIZE, Image.LANCZOS)
                border_color = cmap[g]
                im = ImageOps.expand(im, border=BORDER_THICKNESS, fill=border_color)
                canvas.paste(im, (cx - im.width//2, cy - im.height//2))
            except Exception as e:
                missing += 1
                r = 3
                draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=cmap[g], outline=None)

        out = os.path.join(DIM2_DIR, f"{model}_{method}_可视化.png")
        canvas.save(out, dpi=(150,150))
        logging.info(f"✅ {model} {method} 可视化完成：{out}（缺图占位 {missing}）")

    for key in MODELS:
        if USE_TSNE:
            path = os.path.join(DIM2_DIR, f"{key}_tSNE.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                visualize(df, "TSNE_X", "TSNE_Y", key, "t-SNE")
            else:
                logging.warning(f"⚠️ 未找到 t-SNE 结果：{path}")

        if USE_UMAP:
            path = os.path.join(DIM2_DIR, f"{key}_UMAP.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                visualize(df, "UMAP_X", "UMAP_Y", key, "UMAP")
            else:
                logging.warning(f"⚠️ 未找到 UMAP 结果：{path}")

    logging.info("\n🎯 全部视觉特征分析流程完成！")
