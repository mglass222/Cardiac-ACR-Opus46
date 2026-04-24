#!/usr/bin/env python
# coding: utf-8

"""
TensorDataset wrapper around the cached UNI2-h feature files.

The feature cache is small enough (thousands of patches × 1536 floats
= a few tens of MB per split) to load fully into memory and hand to a
vanilla ``DataLoader``.
"""

import os

import torch
from torch.utils.data import TensorDataset


class FeatureCache:
    """Loaded contents of one encoded split."""

    def __init__(self, features, labels, classes):
        self.features = features    # FloatTensor [N, EMBED_DIM]
        self.labels = labels        # LongTensor  [N]
        self.classes = list(classes)

    @classmethod
    def load(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Feature cache not found: {path}\n"
                "Run `python -m cardiac_acr.backends.uni.encode_patches` first."
            )
        blob = torch.load(path, weights_only=False)
        return cls(blob["features"], blob["labels"], blob["classes"])

    def __len__(self):
        return self.features.size(0)

    def as_tensor_dataset(self):
        return TensorDataset(self.features, self.labels)

    def class_counts(self):
        """Return ``{class_name: count}`` across all patches."""
        counts = {c: 0 for c in self.classes}
        for idx, name in enumerate(self.classes):
            counts[name] = int((self.labels == idx).sum().item())
        return counts
