#!/usr/bin/env python
# coding: utf-8

"""
Encode the 2026 repo's extracted patches with UNI2-h and cache features.

Walks ``cg.TRAIN_DIR`` and ``cg.VALID_DIR`` (PNGs laid out by
``torchvision.datasets.ImageFolder`` convention), runs each image
through the frozen UNI2-h backbone, and writes a single ``.pt`` file
per split to ``data/Features/``.

Output schema (per split)::

    {
        "features": FloatTensor [N, 1536],
        "labels":   LongTensor  [N],
        "classes":  list[str]            # index-aligned with labels
    }

Runs once; downstream training loads the cache and doesn't touch the
backbone again.

Usage:
    python -m cardiac_acr.backends.uni.encode_patches
"""

import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cardiac_acr.backends.uni import config as uni_cfg
from cardiac_acr.backends.uni.backbone import UNIBackbone


def _build_transform():
    """UNI2-h input pipeline: resize to 224, tensor, ImageNet-normalize."""
    return transforms.Compose([
        transforms.Resize(uni_cfg.INPUT_SIZE),
        transforms.CenterCrop(uni_cfg.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(uni_cfg.IMAGENET_MEAN, uni_cfg.IMAGENET_STD),
    ])


def _encode_split(backbone, split_dir, out_path):
    """Encode one ImageFolder-style split and write the feature cache."""
    # allow_empty=True so classes with no images in this split (e.g.
    # Validation/Hemorrhage — only annotated on one training slide)
    # don't crash the walk.
    dataset = datasets.ImageFolder(
        split_dir,
        transform=_build_transform(),
        allow_empty=True,
    )
    classes = dataset.classes

    if len(dataset) == 0:
        print(f"  [{split_dir}] no patches found — writing empty cache")
        torch.save(
            {
                "features": torch.empty(0, uni_cfg.EMBED_DIM, dtype=torch.float32),
                "labels": torch.empty(0, dtype=torch.long),
                "classes": classes,
            },
            out_path,
        )
        return

    loader = DataLoader(
        dataset,
        batch_size=uni_cfg.ENCODE_BATCH_SIZE,
        shuffle=False,
        num_workers=uni_cfg.ENCODE_NUM_WORKERS,
        pin_memory=True,
    )

    all_feats, all_labels = [], []
    total = len(dataset)
    t0 = time.time()

    for i, (images, labels) in enumerate(loader):
        feats = backbone.encode(images)
        all_feats.append(feats)
        all_labels.append(labels)

        done = min((i + 1) * uni_cfg.ENCODE_BATCH_SIZE, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(
            f"  {done:>6}/{total} patches | "
            f"{rate:6.1f} img/s | "
            f"{elapsed:6.1f}s elapsed",
            end="\r",
        )

    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print()
    print(f"  features: {tuple(features.shape)}  labels: {tuple(labels.shape)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {"features": features, "labels": labels, "classes": classes},
        out_path,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  wrote {out_path} ({size_mb:.1f} MB)")


def main():
    print("Loading UNI2-h backbone...")
    backbone = UNIBackbone()
    print(f"  device: {backbone.device}  dtype: {backbone.dtype}")

    for split, src, dst in [
        ("Training",   uni_cfg.TRAIN_DIR, uni_cfg.TRAINING_FEATURES_PATH),
        ("Validation", uni_cfg.VALID_DIR, uni_cfg.VALIDATION_FEATURES_PATH),
    ]:
        print(f"\n=== {split} ({src}) ===")
        if not os.path.isdir(src):
            raise SystemExit(
                f"Source directory not found: {src}\n"
                "Run extract_patches + create_training_sets in the 2026 repo first."
            )
        _encode_split(backbone, src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
