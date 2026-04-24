#!/usr/bin/env python
# coding: utf-8

"""
Encode the 2026 repo's extracted patches with UNI2-h and cache features.

Walks ``cg.TRAIN_DIR`` and ``cg.VALID_DIR`` (PNGs laid out by
``torchvision.datasets.ImageFolder`` convention), runs each image
through the frozen UNI2-h backbone, and writes a single ``.pt`` file
per split to ``data/Features/``.

Training patches are encoded under ``uni_cfg.NUM_TRAIN_VIEWS``
deterministic D4 symmetries (rotations + flips) so the head sees each
patch in multiple orientations — a free augmentation since H&E
histopathology has no preferred orientation. Validation patches are
always encoded once (canonical view) so val-acc stays comparable
across runs.

Output schema (per split)::

    {
        "features": FloatTensor [N * num_views, 1536],
        "labels":   LongTensor  [N * num_views],
        "classes":  list[str]            # index-aligned with labels
    }

Runs once; downstream training loads the cache and doesn't touch the
backbone again.

Usage:
    python -m cardiac_acr.backends.uni.encode_patches
"""

import os
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from cardiac_acr.backends.uni import config as uni_cfg
from cardiac_acr.backends.uni.backbone import UNIBackbone


# The 8 elements of the dihedral group D4: 4 rotations x 2 flips.
# View 0 is the canonical (no rotation, no flip) orientation.
_D4_VIEWS = [
    (0,   False),
    (90,  False),
    (180, False),
    (270, False),
    (0,   True),
    (90,  True),
    (180, True),
    (270, True),
]


def _apply_d4(img, k_rot, do_flip):
    """Apply one fixed D4 symmetry to a PIL image."""
    if do_flip:
        img = TF.hflip(img)
    if k_rot:
        img = TF.rotate(img, k_rot)
    return img


def _build_view_transforms(num_views):
    """Return ``num_views`` deterministic Compose transforms.

    ``num_views == 1`` reproduces the canonical legacy pipeline. Higher
    values map to a prefix of ``_D4_VIEWS`` (in particular, ``num_views
    == 8`` covers the full dihedral group).
    """
    if num_views < 1 or num_views > len(_D4_VIEWS):
        raise ValueError(
            f"num_views must be in [1, {len(_D4_VIEWS)}], got {num_views}"
        )

    pipelines = []
    for k_rot, do_flip in _D4_VIEWS[:num_views]:
        pipelines.append(transforms.Compose([
            transforms.Resize(uni_cfg.INPUT_SIZE),
            transforms.CenterCrop(uni_cfg.INPUT_SIZE),
            transforms.Lambda(partial(_apply_d4, k_rot=k_rot, do_flip=do_flip)),
            transforms.ToTensor(),
            transforms.Normalize(uni_cfg.IMAGENET_MEAN, uni_cfg.IMAGENET_STD),
        ]))
    return pipelines


def _encode_split(backbone, split_dir, out_path, num_views):
    """Encode one ImageFolder-style split under ``num_views`` D4 views.

    The output cache concatenates all views: shape
    ``(num_patches * num_views, EMBED_DIM)``. Labels are duplicated
    per view. ``train.py`` shuffles the loader, so view order in the
    saved cache doesn't matter.
    """
    view_transforms = _build_view_transforms(num_views)

    # Probe the directory once with the canonical transform to decide
    # whether the split is empty and to grab the canonical class list.
    probe = datasets.ImageFolder(
        split_dir,
        transform=view_transforms[0],
        allow_empty=True,
    )
    classes = probe.classes
    num_patches = len(probe)

    if num_patches == 0:
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

    print(f"  {num_patches} patches x {num_views} view(s) "
          f"= {num_patches * num_views} encodings")

    all_feats, all_labels = [], []
    total = num_patches * num_views
    t0 = time.time()
    done = 0

    for view_idx, view_transform in enumerate(view_transforms):
        # Fresh ImageFolder per view so the transform is fixed for the
        # whole pass; reuse the probed class list.
        dataset = datasets.ImageFolder(
            split_dir,
            transform=view_transform,
            allow_empty=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=uni_cfg.ENCODE_BATCH_SIZE,
            shuffle=False,
            num_workers=uni_cfg.ENCODE_NUM_WORKERS,
            pin_memory=True,
        )

        for images, labels in loader:
            feats = backbone.encode(images)
            all_feats.append(feats)
            all_labels.append(labels)

            done += images.size(0)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"  view {view_idx + 1}/{num_views} | "
                f"{done:>7}/{total} encodings | "
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

    splits = [
        ("Training",   uni_cfg.TRAIN_DIR, uni_cfg.TRAINING_FEATURES_PATH,
         uni_cfg.NUM_TRAIN_VIEWS),
        ("Validation", uni_cfg.VALID_DIR, uni_cfg.VALIDATION_FEATURES_PATH,
         1),
    ]
    for split, src, dst, num_views in splits:
        print(f"\n=== {split} ({src}) ===")
        if not os.path.isdir(src):
            raise SystemExit(
                f"Source directory not found: {src}\n"
                "Run extract_patches + create_training_sets in the 2026 repo first."
            )
        _encode_split(backbone, src, dst, num_views)

    print("\nDone.")


if __name__ == "__main__":
    main()
