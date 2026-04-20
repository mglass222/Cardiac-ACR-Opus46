#!/usr/bin/env python
# coding: utf-8

"""
ResNet model factory for cardiac-ACR patch classification.

Swaps the original torchvision ``pretrained=True`` API for the modern
``weights=`` enum to avoid the deprecation warning.

Freezing policy (unchanged from ``Cardiac_ACR_Pytorch_V8_FINAL.ipynb``):
  * Stage 1 (``train_fc_only``): only FC + BatchNorm layers require grad.
  * Stage 2 (``train_unlocked_layers``): the caller additionally unfreezes
    ``layer3`` and ``layer4`` — that is handled in ``train.py``.

The FC head is replaced with ``Dropout(0.5) -> Linear(num_ftrs, num_classes)``.
"""

import torch.nn as nn
from torchvision import models


# Maps friendly model names to (constructor, default-weights-enum) pairs.
# Using the modern ``weights=`` API instead of deprecated ``pretrained=True``.
_RESNET_FACTORIES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2),
}


def build_resnet(model_name, num_classes, dropout_p=0.5):
    """
    Build a pretrained ResNet with a fresh classification head.

    Parameters
    ----------
    model_name : str
        One of the keys in ``_RESNET_FACTORIES``.
    num_classes : int
        Number of output logits (patch classes).
    dropout_p : float, optional
        Dropout probability in the classification head.

    Returns
    -------
    torch.nn.Module
        A ResNet ready for ``train_fc_only``. Only FC + BatchNorm
        parameters have ``requires_grad=True``.
    """
    if model_name not in _RESNET_FACTORIES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Expected one of: {sorted(_RESNET_FACTORIES)}"
        )

    constructor, weights = _RESNET_FACTORIES[model_name]
    model = constructor(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(num_ftrs, num_classes),
    )

    _set_requires_grad_bn_and_fc_only(model)

    print(f"Model = {model_name}")
    print("Model Initialized...")
    return model


def _set_requires_grad_bn_and_fc_only(model):
    """Freeze everything except BatchNorm and FC layers."""
    for name, param in model.named_parameters():
        if "bn" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_layers(model, layer_names=("layer3", "layer4")):
    """
    Switch ``requires_grad`` on for the named top-level ResNet layers.

    Used before the second (fine-tuning) training stage so ``layer3`` /
    ``layer4`` can receive gradient updates alongside the BN + FC head.
    """
    for layer_name in layer_names:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True
    return model
