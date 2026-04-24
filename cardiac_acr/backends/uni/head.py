#!/usr/bin/env python
# coding: utf-8

"""
Classifier heads that sit on top of frozen UNI2-h features.

Two variants:

- ``LinearHead``: a single ``Linear(EMBED_DIM, num_classes)``. The
  canonical UNI evaluation protocol and the recommended default.
- ``MLPHead``: ``Linear(EMBED_DIM, hidden) -> ReLU -> Dropout ->
  Linear(hidden, num_classes)``. Use only if the linear probe
  underperforms.
"""

import torch
import torch.nn as nn

from cardiac_acr.backends.uni import config as uni_cfg


class LinearHead(nn.Module):
    """Single linear layer over UNI2-h CLS-token features."""

    def __init__(self, embed_dim=uni_cfg.EMBED_DIM,
                 num_classes=uni_cfg.NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    """Two-layer MLP — use only if the linear probe underperforms."""

    def __init__(self, embed_dim=uni_cfg.EMBED_DIM,
                 num_classes=uni_cfg.NUM_CLASSES,
                 hidden_dim=uni_cfg.HEAD_HIDDEN_DIM,
                 dropout=uni_cfg.HEAD_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_head(head_type=None, **kwargs):
    """Construct the head named by ``head_type`` (or ``cg.HEAD_TYPE``)."""
    if head_type is None:
        head_type = uni_cfg.HEAD_TYPE
    head_type = head_type.lower()
    if head_type == "linear":
        return LinearHead(**kwargs)
    if head_type == "mlp":
        return MLPHead(**kwargs)
    raise ValueError(f"Unknown head type: {head_type!r} (expected 'linear' or 'mlp')")
