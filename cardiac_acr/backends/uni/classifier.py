#!/usr/bin/env python
# coding: utf-8

"""
UNI2-h ``BackendClassifier`` loader.

Bundles the frozen UNI2-h backbone + trained head + preprocessing
transform + per-backend output paths into the
:class:`cardiac_acr.backends.BackendClassifier` shape consumed by
``wsi/diagnose.py``.
"""

from typing import Optional

import torch
from torchvision import transforms

from cardiac_acr.backends import BackendClassifier
from cardiac_acr.backends.uni import config as uni_cfg
from cardiac_acr.backends.uni.backbone import UNIBackbone
from cardiac_acr.backends.uni.evaluate import load_head_checkpoint


def _make_transform():
    return transforms.Compose([
        transforms.Resize(uni_cfg.INPUT_SIZE),
        transforms.CenterCrop(uni_cfg.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(uni_cfg.IMAGENET_MEAN, uni_cfg.IMAGENET_STD),
    ])


def load_classifier(
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> BackendClassifier:
    backbone = UNIBackbone(device=device)
    head, blob = load_head_checkpoint(path=checkpoint_path)
    head = head.to(device).eval()

    @torch.no_grad()
    def classify(batch_on_device: torch.Tensor) -> torch.Tensor:
        # Backbone runs in bf16 autocast internally and returns float32
        # CPU features; keep that, then move to `device` for the head.
        feats = backbone.encode(batch_on_device.cpu())
        return head(feats.to(device))

    return BackendClassifier(
        name="uni",
        classify=classify,
        classes=list(blob["classes"]),
        transform=_make_transform(),
        device=device,
        saved_database_dir=uni_cfg.SAVED_DATABASE_DIR,
        slide_dx_dir=uni_cfg.SLIDE_DX_DIR,
        annotated_png_dir=uni_cfg.ANNOTATED_PNG_DIR,
        test_slide_predictions_dir=uni_cfg.TEST_SLIDE_PREDICTIONS_DIR,
        test_slide_annotations_dir=uni_cfg.TEST_SLIDE_ANNOTATIONS_DIR,
    )
