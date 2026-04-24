#!/usr/bin/env python
# coding: utf-8

"""
Evaluate the trained UNI2-h head on the validation feature cache.

Produces:
  - per-class precision / recall / F1
  - macro + weighted averages
  - one-vs-rest AUROC per class
  - confusion matrix

Same metric shape as the 2026 ResNet-50 baseline stats scripts, so the
numbers line up head-to-head.

Usage:
    python -m cardiac_acr.backends.uni.evaluate
"""

import os

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from cardiac_acr.backends.uni import config as uni_cfg
from cardiac_acr.backends.uni.features_dataset import FeatureCache
from cardiac_acr.backends.uni.head import build_head


def load_head_checkpoint(path=None):
    """Load a trained head checkpoint written by ``train.py``."""
    if path is None:
        path = os.path.join(
            uni_cfg.MODEL_DIR, f"uni2h_{uni_cfg.HEAD_TYPE}_head.pt"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No head checkpoint at {path}. Run `python -m cardiac_acr.backends.uni.train`."
        )
    blob = torch.load(path, weights_only=False, map_location="cpu")
    model = build_head(blob["head_type"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def _predict(model, features, device):
    """Run the head over all cached features; return (logits, probs, preds)."""
    model = model.to(device)
    features = features.to(device)
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
    return logits.cpu(), probs.cpu(), preds.cpu()


def _one_vs_rest_auroc(labels, probs, classes):
    """Per-class one-vs-rest AUROC. Classes with <2 label values get NaN."""
    y_true = labels.numpy()
    scores = probs.numpy()
    out = {}
    for idx, name in enumerate(classes):
        positives = (y_true == idx)
        if positives.sum() == 0 or positives.sum() == len(y_true):
            out[name] = float("nan")
            continue
        out[name] = float(roc_auc_score(positives, scores[:, idx]))
    return out


def evaluate(head_path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_cache = FeatureCache.load(uni_cfg.VALIDATION_FEATURES_PATH)
    if len(val_cache) == 0:
        raise SystemExit(
            "Validation feature cache is empty — nothing to evaluate."
        )
    print(f"Validation patches: {len(val_cache)}")
    print(f"Class counts: {val_cache.class_counts()}")

    model, blob = load_head_checkpoint(head_path)
    print(f"Loaded head: {blob['head_type']} ({sum(p.numel() for p in model.parameters())} params)")

    logits, probs, preds = _predict(model, val_cache.features, device)
    labels = val_cache.labels
    classes = val_cache.classes

    print("\n=== Per-class report ===")
    # labels=range(...) so empty classes still appear in the table.
    print(classification_report(
        labels.numpy(),
        preds.numpy(),
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    ))

    print("=== One-vs-rest AUROC ===")
    auroc = _one_vs_rest_auroc(labels, probs, classes)
    for name, value in auroc.items():
        print(f"  {name:<12} {value:.4f}" if not np.isnan(value)
              else f"  {name:<12} n/a (single-label class)")
    valid = [v for v in auroc.values() if not np.isnan(v)]
    if valid:
        print(f"  macro-avg    {np.mean(valid):.4f}  (over {len(valid)} classes)")

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(
        labels.numpy(), preds.numpy(), labels=list(range(len(classes)))
    )
    col_headers = " ".join(f"{c:>10}" for c in classes)
    print(f"{'':<12}{col_headers}")
    for i, row in enumerate(cm):
        cells = " ".join(f"{v:>10d}" for v in row)
        print(f"{classes[i]:<12}{cells}")


def main():
    evaluate()


if __name__ == "__main__":
    main()
