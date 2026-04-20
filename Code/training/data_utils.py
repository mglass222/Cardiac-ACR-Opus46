#!/usr/bin/env python
# coding: utf-8

"""
Dataset statistics and dataloader construction for training.

This module owns everything that touches the on-disk patch library
produced by ``extract_patches.py`` — counting classes/patches, computing
class weights and dataset statistics, and wiring up the training /
validation ``DataLoader`` pair.

Ported from the ``Cardiac_ACR_Pytorch_V8_FINAL.ipynb`` notebook. Paths
that were hard-coded to ``D:\\Cardiac_ACR\\...`` now flow through
``cardiac_globals``.
"""

import os
import sys
from os import listdir
from os.path import isdir, join

import cv2
import torch
from torchvision import datasets, transforms

# Allow imports from the parent ``Code/`` directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cardiac_globals as cg  # noqa: E402


def count_classes(train_dir=None):
    """Return the number of class subdirectories under ``train_dir``."""
    if train_dir is None:
        train_dir = cg.TRAIN_DIR

    dirs = [d for d in listdir(train_dir) if isdir(join(train_dir, d))]
    num_classes = len(dirs)
    print("number of classes =", num_classes)
    return num_classes


def count_patches(train_dir=None, valid_dir=None):
    """Return ``(train_patches, valid_patches)`` file counts."""
    if train_dir is None:
        train_dir = cg.TRAIN_DIR
    if valid_dir is None:
        valid_dir = cg.VALID_DIR

    train_patches = sum(
        len(listdir(join(train_dir, d))) for d in listdir(train_dir)
    )
    valid_patches = sum(
        len(listdir(join(valid_dir, d))) for d in listdir(valid_dir)
    )
    return train_patches, valid_patches


def epoch_steps(batch_size, train_dir=None, valid_dir=None):
    """Compute training / validation steps per epoch for logging."""
    train_patches, valid_patches = count_patches(train_dir, valid_dir)
    train_steps = round(train_patches / batch_size)
    valid_steps = round(valid_patches / batch_size)

    print("Number of training pics:", train_patches)
    print("Number of validation pics:", valid_patches)
    print("batch size =", batch_size)
    print("train steps per epoch:", train_steps)
    print("valid steps per epoch:", valid_steps)

    return train_steps, valid_steps


def class_weights(train_dir=None):
    """
    Compute balanced (sklearn-style) per-class loss weights.

    weight_i = total_train_patches / (num_classes * patches_in_class_i)
    """
    if train_dir is None:
        train_dir = cg.TRAIN_DIR

    num_classes = count_classes(train_dir)
    num_train_patches, _ = count_patches(train_dir=train_dir)

    weights = []
    for d in sorted(listdir(train_dir)):
        class_path = join(train_dir, d)
        if not isdir(class_path):
            continue
        num_class_images = len(listdir(class_path))
        weights.append(num_train_patches / (num_classes * num_class_images))

    return weights


def get_percentages(train_dir=None, valid_dir=None, openslide_dir=None):
    """
    Fraction of each class that landed in the training split.

    Returns ``{class_name: percent_train}`` rounded to two decimals.
    """
    if train_dir is None:
        train_dir = cg.TRAIN_DIR
    if valid_dir is None:
        valid_dir = cg.VALID_DIR
    if openslide_dir is None:
        openslide_dir = cg.OPENSLIDE_DIR

    percent_dict = {}
    classes = listdir(openslide_dir)

    for item in classes:
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


def dataset_normalization(train_dir=None):
    """
    Compute per-channel mean/std over every patch in ``train_dir``.

    Cached values from the original notebook for reference:
      mean = [0.7109, 0.6954, 0.5749]
      std  = [0.1917, 0.2091, 0.2144]

    The default pipeline uses ImageNet normalization
    (``cg.IMAGENET_MEAN`` / ``cg.IMAGENET_STD``) — this helper is
    provided for experimentation.
    """
    if train_dir is None:
        train_dir = cg.TRAIN_DIR

    print("Getting normalization data...")
    to_tensor = transforms.ToTensor()
    tensor_array = []

    for d in listdir(train_dir):
        directory = join(train_dir, d)
        if not isdir(directory):
            continue
        for image_name in listdir(directory):
            im = cv2.imread(join(directory, image_name))
            im = cv2.resize(im, (cg.TRAIN_INPUT_SIZE, cg.TRAIN_INPUT_SIZE))
            tensor_array.append(to_tensor(im))

    tensor_array = torch.stack(tensor_array)
    mean = tensor_array.view(3, -1).mean(dim=1)
    std = tensor_array.view(3, -1).std(dim=1)

    print("mean =", mean, "std =", std)
    return mean, std


def initialize_dataloaders(
    input_size=None,
    batch_size=None,
    training_root=None,
    num_workers=16,
    pin_memory=True,
):
    """
    Build the training/validation ``DataLoader`` pair.

    The training transform applies color jitter and random rotation on
    top of an ImageNet-normalized tensor; the validation transform only
    normalizes.
    """
    if input_size is None:
        input_size = cg.TRAIN_INPUT_SIZE
    if batch_size is None:
        batch_size = cg.TRAIN_BATCH_SIZE
    if training_root is None:
        training_root = cg.TRAINING_PATCH_DIR

    mean = cg.IMAGENET_MEAN
    std = cg.IMAGENET_STD

    data_transforms = {
        "Training": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ColorJitter(
                brightness=(0.8, 1.5),
                contrast=(0.4, 1.0),
                saturation=0.5,
                hue=(-0.1, 0.1),
            ),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "Validation": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }

    image_datasets = {
        phase: datasets.ImageFolder(
            os.path.join(training_root, phase), data_transforms[phase]
        )
        for phase in ("Training", "Validation")
    }

    dataloaders = {
        phase: torch.utils.data.DataLoader(
            image_datasets[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for phase in ("Training", "Validation")
    }

    print("Initializing Datasets and Dataloaders...\n")
    return dataloaders
