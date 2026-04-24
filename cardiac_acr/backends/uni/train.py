#!/usr/bin/env python
# coding: utf-8

"""
Train the classifier head on cached UNI2-h features.

Loads the feature caches produced by ``encode_patches.py`` and trains a
``LinearHead`` (or ``MLPHead``) with AdamW + cosine schedule + linear
warmup, class-balanced cross-entropy, and best-validation checkpointing.

Fast — features are already in memory, so epochs take seconds. GPU is
optional but used when available.

Usage:
    python -m cardiac_acr.backends.uni.train
"""

import copy
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cardiac_acr.backends.uni import config as uni_cfg
from cardiac_acr.backends.uni.features_dataset import FeatureCache
from cardiac_acr.backends.uni.head import build_head


def _class_weights(labels, num_classes):
    """Balanced per-class loss weights, sklearn-style.

    weight_i = total / (num_classes * count_i)

    Classes with zero patches get weight 0 to avoid div-by-zero (they
    contribute nothing to the loss, which is what we want).
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    weights = torch.zeros(num_classes)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return weights


def _cosine_with_warmup(step, total_steps, warmup_steps):
    """Linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_head(
    head_type=None,
    lr=None,
    weight_decay=None,
    num_epochs=None,
    batch_size=None,
    warmup_epochs=None,
    device=None,
):
    head_type = head_type or uni_cfg.HEAD_TYPE
    lr = lr if lr is not None else uni_cfg.TRAIN_LEARNING_RATE
    weight_decay = weight_decay if weight_decay is not None else uni_cfg.TRAIN_WEIGHT_DECAY
    num_epochs = num_epochs if num_epochs is not None else uni_cfg.TRAIN_NUM_EPOCHS
    batch_size = batch_size if batch_size is not None else uni_cfg.TRAIN_BATCH_SIZE
    warmup_epochs = warmup_epochs if warmup_epochs is not None else uni_cfg.TRAIN_COSINE_WARMUP_EPOCHS
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cache = FeatureCache.load(uni_cfg.TRAINING_FEATURES_PATH)
    val_cache = FeatureCache.load(uni_cfg.VALIDATION_FEATURES_PATH)

    print("Training patches per class:", train_cache.class_counts())
    print("Validation patches per class:", val_cache.class_counts())

    num_classes = len(train_cache.classes)
    weights = _class_weights(train_cache.labels, num_classes).to(device)
    print("class weights:", weights.cpu().tolist())

    model = build_head(head_type).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(
        train_cache.as_tensor_dataset(),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_cache.as_tensor_dataset(),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    print(f"\nHead: {head_type} | epochs: {num_epochs} | batch: {batch_size} | "
          f"lr: {lr} | device: {device}")

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    step = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for feats, labels in train_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            scale = _cosine_with_warmup(step, total_steps, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr * scale

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * feats.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += feats.size(0)
            step += 1

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(feats)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * feats.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += feats.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"epoch {epoch+1:>3}/{num_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {lr * scale:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s. Best val acc: {best_acc:.4f}")

    model.load_state_dict(best_state)
    _save_checkpoint(model, head_type, train_cache.classes)
    return model


def _save_checkpoint(model, head_type, classes):
    os.makedirs(uni_cfg.MODEL_DIR, exist_ok=True)
    path = os.path.join(uni_cfg.MODEL_DIR, f"uni2h_{head_type}_head.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "head_type": head_type,
            "classes": classes,
            "embed_dim": uni_cfg.EMBED_DIM,
        },
        path,
    )
    print(f"Saved head checkpoint -> {path}")


def main():
    train_head()


if __name__ == "__main__":
    main()
