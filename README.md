# 从端正到扭曲：不同书写目标下的符号变形规律与 AI 识别局限

本仓库整理“智能媒体工作坊”选题 4 与选题 5 的代码、结果输出和报告材料。项目没有从零重写课程案例，而是复用 `基础案例A-视觉分析2/视觉分析.py` 的核心流程：`Data` 子文件夹分组、图像索引、视觉特征提取、PCA、t-SNE 和可视化。在此基础上增加本地 `torchvision` 模型、形态学特征、分类检验和统一输出目录。

## 选题关系

选题 4 用三类书写目标证明：端正易读、美观、极致扭曲三类符号图像在深度视觉特征空间中存在可测量差异。

选题 5 是选题 4 的升级拓展：选题 4 显示“美观 vs 极致扭曲”比“端正 vs 扭曲”更容易重叠，因此选题 5 聚焦这组细粒度二分类，比较 ResNet50、Swin_T、Morphology 和 Fusion 特征，分析 AI 对符号变形规律的识别能力及局限。

## 项目结构

```text
F:\project\智能媒体工作坊
├─ 基础案例A-视觉分析2\
│  ├─ 视觉分析.py
│  ├─ 视觉分析_backup.py
│  ├─ 视觉分析_选题4_ResNet50.py
│  └─ Data\端正易读|美观|极致扭曲
├─ 视觉特征计算\model\
├─ 选题5_基于题4拓展分析\
├─ src\
├─ outputs\topic4
├─ outputs\topic5
├─ docs\
├─ requirements.txt
└─ README.md
```

## 环境要求

建议使用 Windows + Anaconda/PowerShell。依赖见 `requirements.txt`：

```powershell
pip install -r requirements.txt
```

## 本地模型

脚本优先检查并使用本地模型，不强制联网下载：

- `视觉特征计算\model\torchvision\ResNet50.pth`
- `视觉特征计算\model\torchvision\Swin_T.pth`

如果 `.pth` 权重不存在，`src/feature_extraction.py` 会 fallback 到 `weights=None`。随机初始化只能验证流程，不适合作为正式实验结论。

检查资源：

```powershell
python src/main.py --task check
```

## 数据裁切

为精简提交内容，仓库不保留原始扫描页，只保留 `基础案例A-视觉分析2/Data` 中的 8640 张裁切图像和 `image_index.csv`。

如果需要重新裁切，应先把三类原始扫描页恢复到项目根目录，再运行：

```powershell
python src/crop_pages.py
```

默认配置为 `config/crop_config.json`。已有裁切图片时脚本只重建 `Data/image_index.csv`，不会覆盖图片；需要重新裁切时使用：

```powershell
python src/crop_pages.py --force
```

## 运行选题 4 / 选题 5

运行全部整理流程：

```powershell
python src/main.py --all
```

只运行选题 4：

```powershell
python src/main.py --task topic4
```

只运行选题 5：

```powershell
python src/main.py --task topic5
```

如果已经有 PCA 和缓存特征，不想重跑深度特征：

```powershell
python src/main.py --all --skip-feature
```

## 输出结果

选题 4 输出：

- `outputs/topic4/figures/topic4_sample_examples.png`
- `outputs/topic4/figures/topic4_resnet50_pca_scatter.png`
- `outputs/topic4/figures/topic4_resnet50_tsne_scatter.png`
- `outputs/topic4/figures/topic4_confusion_matrix.png`
- `outputs/topic4/tables/topic4_group_distances.csv`
- `outputs/topic4/metrics/topic4_classification_report.txt`

选题 5 输出：

- `outputs/topic5/figures/topic5_accuracy_macro_f1_bars.png`
- `outputs/topic5/figures/topic5_feature_space_comparison.png`
- `outputs/topic5/figures/topic5_confusion_matrices.png`
- `outputs/topic5/figures/topic5_fusion_top15_feature_importance.png`
- `outputs/topic5/figures/topic5_misclassified_examples.png`
- `outputs/topic5/tables/topic5_model_performance.csv`
- `outputs/topic5/tables/topic5_fusion_feature_importance.csv`
- `outputs/topic5/tables/topic5_misclassified_samples.csv`

## 常见问题

1. `ModuleNotFoundError: pandas`  
   运行 `pip install -r requirements.txt`。

2. `ViTFeatureExtractor` 过时  
   原始 `视觉分析.py` 中仍有旧接口，因此本项目正式实验优先使用 `torchvision` 的 ResNet50 和 Swin_T 本地权重。

3. GitHub 上传过大  
   `.gitignore` 已忽略原始扫描页、裁切数据和大模型权重。若大文件已进入旧提交，仅删除工作区文件不会缩小 `.git`，还需要另行重建或清理 Git 历史。
