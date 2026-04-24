#!/usr/bin/env python
# coding: utf-8

"""
Patch-library count helpers used by ``create_training_sets``.

The original Cardiac-ACR codebase used this module as a catch-all for
patch-library statistics, dataset normalization, and DataLoader
construction. In the UNI pipeline the head trains on cached features
instead of raw patches, so only the count helpers are still useful —
everything DataLoader-shaped now lives in ``features_dataset.py``.
"""

from os import listdir
from os.path import isdir, join

from cardiac_acr import config as cg


def count_classes(train_dir=None):
    """Return the number of class subdirectories under ``train_dir``."""
    if train_dir is None:
        train_dir = cg.TRAIN_DIR
    dirs = [d for d in listdir(train_dir) if isdir(join(train_dir, d))]
    return len(dirs)


def count_patches(train_dir=None, valid_dir=None):
    """Return ``(train_patches, valid_patches)`` file counts."""
    if train_dir is None:
        train_dir = cg.TRAIN_DIR
    if valid_dir is None:
        valid_dir = cg.VALID_DIR
    train_patches = sum(len(listdir(join(train_dir, d))) for d in listdir(train_dir))
    valid_patches = sum(len(listdir(join(valid_dir, d))) for d in listdir(valid_dir))
    return train_patches, valid_patches


def get_percentages(train_dir=None, valid_dir=None, openslide_dir=None):
    """Fraction of each class's patches that landed in the training split.

    Returns ``{class_name: percent_train}`` rounded to two decimals.
    Called from ``create_training_sets.main()`` as a post-split sanity
    check on the per-class split ratios.
    """
    if train_dir is None:
        train_dir = cg.TRAIN_DIR
    if valid_dir is None:
        valid_dir = cg.VALID_DIR
    if openslide_dir is None:
        openslide_dir = cg.OPENSLIDE_DIR

    percent_dict = {}
    for item in listdir(openslide_dir):
        train_path = join(train_dir, item)
        val_path = join(valid_dir, item)
        if isdir(train_path) and isdir(val_path):
            num_train = len(listdir(train_path))
            num_val = len(listdir(val_path))
            total = num_train + num_val
            if total == 0:
                continue
            percent_dict[item] = round(num_train / total, 2)

    return percent_dict
