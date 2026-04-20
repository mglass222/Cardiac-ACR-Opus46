#!/usr/bin/env python
# coding: utf-8

"""
Patch-level statistics for the training set.

Consumes ``cg.TRAIN_SET_PREDICTIONS_PICKLE`` (produced by
:mod:`dump_training_predictions`) and produces:

1. A 6-class confusion matrix over every training patch, with axis
   labels from :data:`cardiac_globals.CLASS_NAMES`.
2. A 2x2 grid of one-vs-rest ROC curves for the four clinically
   interesting classes: ``1R1A``, ``1R2``, ``Healing``, ``Normal``.

Ported from the upper half of ``Cardiac_ACR_Pytorch_Training_Set_Stats_V6.ipynb``
(cells 17-22). The original notebook mapped classes manually with a
dict that included a typo (``'Hemmorhage'``); this module reuses the
canonical :data:`cardiac_globals.CLASS_TO_INDEX` map instead.
"""

import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from cardiac_acr import cardiac_globals as cg


# Classes for which we plot per-class ROC curves. ``Hemorrhage`` and
# ``Quilty`` are excluded — they have too few patches to yield a
# meaningful curve in practice.
ROC_CLASSES = ["1R1A", "1R2", "Healing", "Normal"]


def load_predictions(pickle_path=None):
    """
    Load the ``[label, softmax_probs]`` pairs persisted by
    :mod:`dump_training_predictions`. Returns ``(labels, predictions)``
    where ``predictions`` is the raw list of pickle entries.
    """
    if pickle_path is None:
        pickle_path = cg.TRAIN_SET_PREDICTIONS_PICKLE

    with open(pickle_path, "rb") as handle:
        predictions = pickle.load(handle)

    labels = [entry[0] for entry in predictions]
    return labels, predictions


def binarize_labels(labels, pos_class):
    """
    One-vs-rest binarization: return 1 for ``pos_class``, 0 otherwise.

    ``pos_class`` must be a key of :data:`cardiac_globals.CLASS_TO_INDEX`.
    """
    if pos_class not in cg.CLASS_TO_INDEX:
        raise ValueError(
            f"Unknown class {pos_class!r}; expected one of "
            f"{sorted(cg.CLASS_TO_INDEX)}"
        )
    pos_idx = cg.CLASS_TO_INDEX[pos_class]
    return [1 if label == pos_idx else 0 for label in labels]


def get_probabilities(predictions, pos_class):
    """
    Extract the per-patch softmax probability of ``pos_class`` across
    every entry in ``predictions``.
    """
    if pos_class not in cg.CLASS_TO_INDEX:
        raise ValueError(
            f"Unknown class {pos_class!r}; expected one of "
            f"{sorted(cg.CLASS_TO_INDEX)}"
        )
    pos_idx = cg.CLASS_TO_INDEX[pos_class]
    return [entry[1][pos_idx] for entry in predictions]


def draw_confusion_matrix(labels, predictions, class_names=None):
    """
    Plot the 6-class patch-level confusion matrix.

    The argmax of each softmax vector is treated as the predicted class.
    Returns the sklearn confusion matrix array.
    """
    if class_names is None:
        class_names = cg.CLASS_NAMES

    preds = [int(np.argmax(entry[1])) for entry in predictions]

    conf_mtx = confusion_matrix(labels, preds)
    accuracy = accuracy_score(labels, preds)
    print(f"Patch-level accuracy = {accuracy:.4f}")

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot()
    cax = ax.matshow(conf_mtx, cmap="Blues")
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(conf_mtx):
        ax.text(
            j, i, f"{z:0.1f}",
            ha="center", va="center", size=15,
            bbox=dict(facecolor="w", edgecolor="black",
                      boxstyle="round,pad=0.5"),
        )

    # Pad with a leading empty label because matshow's default tick
    # positions leave the first tick at index -1.
    ax.set_xticklabels([""] + list(class_names), size=15)
    ax.set_yticklabels([""] + list(class_names), size=15)
    plt.show()

    return conf_mtx


def draw_roc_curves(labels, predictions, classes=None):
    """
    Plot a 2x2 grid of one-vs-rest ROC curves for ``classes``.

    ``classes`` defaults to :data:`ROC_CLASSES`. Returns a dict mapping
    each class name to its AUC score.
    """
    if classes is None:
        classes = ROC_CLASSES

    roc_aucs = {}
    fprs = {}
    tprs = {}
    for pos_class in classes:
        labels_bin = binarize_labels(labels, pos_class)
        class_probs = get_probabilities(predictions, pos_class)
        roc_aucs[pos_class] = roc_auc_score(labels_bin, class_probs)
        fpr, tpr, _ = roc_curve(labels_bin, class_probs)
        fprs[pos_class] = fpr
        tprs[pos_class] = tpr

    num_cols = 2
    num_rows = 2
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.set_size_inches(10, 10)
    fig.tight_layout(pad=5)

    for count, pos_class in enumerate(classes):
        if count >= num_rows * num_cols:
            break
        x, y = divmod(count, num_cols)
        axs[x, y].set_title(pos_class, size=15)
        axs[x, y].plot(
            fprs[pos_class], tprs[pos_class],
            color="blue", linewidth=3,
            label="ROC curve (area = %0.4f)" % roc_aucs[pos_class],
        )
        axs[x, y].plot([0, 1], [0, 1], "k--")
        axs[x, y].axis([-0.01, 1, 0, 1.01])
        axs[x, y].set_xlabel("FPR", size=15)
        axs[x, y].set_ylabel("TPR", size=15)
        axs[x, y].legend(loc="lower right")

    plt.show()

    for pos_class, auc in roc_aucs.items():
        print(f"{pos_class} ROC AUC = {auc:.4f}")

    return roc_aucs


def main():
    labels, predictions = load_predictions()
    print(f"Loaded {len(predictions)} patch predictions")
    draw_confusion_matrix(labels, predictions)
    draw_roc_curves(labels, predictions)


if __name__ == "__main__":
    main()
