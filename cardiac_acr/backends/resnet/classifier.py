#!/usr/bin/env python
# coding: utf-8

"""
ResNet-50 ``BackendClassifier`` loader.

The ResNet pipeline historically saves the entire model via
``torch.save(model, path)`` (see ``backends/resnet/train.py``) rather
than a state_dict. We load it back the same way.
"""

import os
from typing import Optional

import torch
from torchvision import transforms

from cardiac_acr.backends import BackendClassifier
from cardiac_acr.backends.resnet import config as cg


_DEFAULT_CHECKPOINT = os.path.join(cg.MODEL_DIR, "resnet50_ft")


def _make_transform():
    return transforms.Compose([
        transforms.Resize(cg.ANNOTATION_SIZE),
        transforms.CenterCrop(cg.ANNOTATION_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(cg.IMAGENET_MEAN, cg.IMAGENET_STD),
    ])


def load_classifier(
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> BackendClassifier:
    path = checkpoint_path or _DEFAULT_CHECKPOINT
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No ResNet checkpoint at {path}. Train one with "
            "`python -m cardiac_acr.backends.resnet.train` or pass "
            "--checkpoint."
        )

    # ResNet checkpoints in this project are full pickled model objects.
    model = torch.load(path, map_location=device, weights_only=False)
    model = model.to(device).eval()

    @torch.no_grad()
    def classify(batch_on_device: torch.Tensor) -> torch.Tensor:
        return model(batch_on_device)

    return BackendClassifier(
        name="resnet",
        classify=classify,
        classes=list(cg.CLASS_NAMES),
        transform=_make_transform(),
        device=device,
        saved_database_dir=cg.SAVED_DATABASE_DIR,
        slide_dx_dir=cg.SLIDE_DX_DIR,
        annotated_png_dir=cg.ANNOTATED_PNG_DIR,
        test_slide_predictions_dir=cg.TEST_SLIDE_PREDICTIONS_DIR,
        test_slide_annotations_dir=cg.TEST_SLIDE_ANNOTATIONS_DIR,
    )
