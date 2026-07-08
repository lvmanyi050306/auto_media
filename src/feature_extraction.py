# -*- coding: utf-8 -*-
"""Offline-first torchvision feature extraction.

This module does not download models. It first tries local checkpoints under
视觉特征计算/model/torchvision and falls back to random initialization only so
the pipeline can be smoke-tested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import DATA_DIR, PROJECT_ROOT, add_local_pydeps, list_data_images, scan_model_resources

add_local_pydeps()

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms


PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _state_dict(raw):
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    if isinstance(raw, dict):
        return {k.replace("module.", ""): v for k, v in raw.items()}
    return raw


def load_torchvision_model(model_name: str):
    resources = scan_model_resources()
    model_name = model_name.lower()
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        weight_path = Path(resources["ResNet50"])
        feature_dim = 2048
        model.fc = torch.nn.Identity()
    elif model_name in {"swin_t", "swin"}:
        model = models.swin_t(weights=None)
        weight_path = Path(resources["Swin_T"])
        feature_dim = 768
        model.head = torch.nn.Identity()
    elif model_name in {"vit_b16", "vit"}:
        model = models.vit_b_16(weights=None)
        weight_path = Path(resources["ViT_B16"])
        feature_dim = 768
        model.heads = torch.nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    used_weight = "random_init"
    if weight_path.exists():
        raw = torch.load(weight_path, map_location="cpu")
        state = _state_dict(raw)
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            used_weight = str(weight_path)
            if missing or unexpected:
                print(f"权重已加载，但存在不完全匹配: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as exc:
            print(f"本地权重格式不匹配，改用随机初始化: {weight_path} ({exc})")
    else:
        print(f"未找到本地权重，改用随机初始化: {weight_path}")

    model.eval()
    return model, feature_dim, used_weight


def extract_features(model_name: str, out_csv: Path, batch_size: int = 64) -> Path:
    rows = list_data_images(DATA_DIR)
    if not rows:
        raise RuntimeError(f"未在 {DATA_DIR} 下找到裁切图片")
    df = pd.DataFrame(rows)
    model, feature_dim, used_weight = load_torchvision_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    feats = []
    paths = [Path(p) for p in df["image_path"]]
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([PREPROCESS(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            arr = model(batch).detach().cpu().numpy().reshape(len(batch_paths), -1)
        feats.append(arr.astype(np.float32))
        print(f"{model_name}: {min(start + batch_size, len(paths))}/{len(paths)}")

    feat = np.vstack(feats)
    if feat.shape[1] != feature_dim:
        print(f"提示: 预期维度 {feature_dim}, 实际维度 {feat.shape[1]}")
    out = pd.DataFrame(feat, columns=[f"f{i + 1}" for i in range(feat.shape[1])])
    out.insert(0, "group", df["label"].values)
    out.insert(0, "filename", df["filename"].values)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"features: {out_csv}")
    print(f"weight: {used_weight}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "swin_t", "vit_b16"], default="resnet50")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or PROJECT_ROOT / "outputs" / "features" / f"{args.model}_features.csv"
    extract_features(args.model, out)


if __name__ == "__main__":
    main()
