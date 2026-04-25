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
    head, blob = load_head_checkpoint(path=checkpoint_path)
    head = head.to(device).eval()

    # If the checkpoint was produced by the LoRA fine-tune, re-apply
    # the same wrappers to the backbone before loading adapter weights.
    # `compile=False` is required: torch.compile cannot survive the
    # submodule replacement that LoRA performs.
    lora_cfg = blob.get("lora_config")
    if lora_cfg is not None:
        from cardiac_acr.backends.uni.lora import apply_lora_to_uni
        backbone = UNIBackbone(device=device, compile=False)
        apply_lora_to_uni(
            backbone,
            target_blocks=lora_cfg["target_blocks"],
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg.get("dropout", 0.0),
            targets=tuple(lora_cfg.get("targets", ("qkv",))),
        )
        # strict=False because we deliberately don't persist the frozen
        # base UNI2-h weights — those load fresh from HF Hub. Any
        # *unexpected* key (one we don't have a slot for) is a real
        # config mismatch and should fail loudly.
        missing, unexpected = backbone.model.load_state_dict(
            blob["lora_state_dict"], strict=False
        )
        assert not unexpected, f"Unexpected LoRA keys in checkpoint: {unexpected}"
        backbone.model.eval()
    else:
        backbone = UNIBackbone(device=device)

    @torch.no_grad()
    def classify(batch_on_device: torch.Tensor) -> torch.Tensor:
        # Backbone runs in autocast internally and returns float32
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
