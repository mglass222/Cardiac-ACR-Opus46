#!/usr/bin/env python
# coding: utf-8

"""
Slide-level statistics for the test set.

For each confidence-threshold sweep performed by the slide-classifier,
this script:

1. Merges the per-threshold NN diagnosis CSVs under
   ``cg.TEST_SET_ANALYSIS_DIR`` with the pathologist ground-truth CSV
   at ``cg.TEST_DX_CSV`` into a single summary at
   ``cg.TEST_SUMMARY_CSV``.
2. Filters out slides with unknown pathologist diagnosis. Unlike the
   training-set analysis, the test-set analysis keeps AMR slides.
3. Plots a grid of binary (2R vs not-2R) confusion matrices, one per
   threshold.
4. Plots a single 2R-vs-not-2R ROC curve using the sum of per-patch
   ``1R2`` probabilities on each slide as the score.
5. Plots a 4-class confusion matrix (``0R``, ``1R1A``, ``1R2``, ``2R``)
   using the best-accuracy threshold column.

Ported from ``Cardiac_ACR_Pytorch_Test_Set_Stats_V5.ipynb``. All
duplicated helpers live in :mod:`_stats_utils`.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from cardiac_acr import cardiac_globals as cg
from cardiac_acr.stats import _stats_utils as stats_utils


# 4-class confusion-matrix labels in display order.
FOUR_CLASS_LABELS = ["0R", "1R1A", "1R2", "2R"]


def _to_four_class_index(dx):
    """
    Collapse any diagnosis string onto the 4-class {0R, 1R1A, 1R2, 2R}
    axis. Returns ``None`` for unknown / unsupported labels so the
    caller can skip them.
    """
    dx = dx.strip()
    if dx == "0R":
        return 0
    if dx in {"1R1A", "1R1B", "1R"}:
        return 1
    if dx == "1R2":
        return 2
    if dx in {"2R", "2R3A", "3R"}:
        return 3
    return None


def draw_four_class_confusion(path_dx_col, nn_dx_col, title="Test Set"):
    """
    Plot a 4-class (``0R``, ``1R1A``, ``1R2``, ``2R``) confusion matrix.

    ``path_dx_col`` is the pathologist ground-truth column, ``nn_dx_col``
    is the NN diagnosis column from one specific threshold. Rows where
    either label falls outside the 4-class axis are dropped.
    """
    path_int = []
    nn_int = []
    for path_dx, nn_dx in zip(path_dx_col, nn_dx_col):
        p = _to_four_class_index(path_dx)
        n = _to_four_class_index(nn_dx)
        if p is None or n is None:
            continue
        path_int.append(p)
        nn_int.append(n)

    conf_mtx = confusion_matrix(
        path_int, nn_int, labels=list(range(len(FOUR_CLASS_LABELS))),
    )
    accuracy = accuracy_score(path_int, nn_int)
    print(f"{title} 4-class accuracy = {accuracy:.4f}")

    fig, ax = plt.subplots()
    ax.matshow(conf_mtx)
    for (i, j), z in np.ndenumerate(conf_mtx):
        ax.text(j, i, f"{z:0.1f}", ha="center", va="center")

    ax.set_xticklabels([""] + FOUR_CLASS_LABELS)
    ax.set_yticklabels([""] + FOUR_CLASS_LABELS)
    ax.set_title(title)
    plt.show()

    return conf_mtx


def _best_threshold_column(all_results, file_dict):
    """
    Pick the threshold whose 2R-vs-not-2R accuracy matches the
    pathologist column best. Returns ``(key, nn_dx_column)``.
    """
    percent_list = list(file_dict.keys())
    path_dx_bin = stats_utils.convert_to_binary(all_results[0])
    nn_results_bin = [stats_utils.convert_to_binary(col)
                      for col in all_results[1:]]

    accuracies = [accuracy_score(path_dx_bin, col) for col in nn_results_bin]
    best_idx = int(np.argmax(accuracies))
    return percent_list[best_idx], all_results[best_idx + 1]


def main():
    file_dict = stats_utils.get_dx_files(cg.TEST_SET_ANALYSIS_DIR)
    if not file_dict:
        raise FileNotFoundError(
            f"No per-threshold diagnosis CSVs found in "
            f"{cg.TEST_SET_ANALYSIS_DIR}"
        )

    all_results, unknown_slides, slide_list = stats_utils.make_summary_csv(
        cg.TEST_SUMMARY_CSV, cg.TEST_DX_CSV, file_dict,
    )
    all_results = stats_utils.add_results_to_csv(
        cg.TEST_SUMMARY_CSV, file_dict, all_results,
    )
    all_results, _ = stats_utils.filter_csv(
        all_results, slide_list, amr_cutoff=None,
    )

    stats_utils.draw_confusion_mtx(
        all_results, file_dict, title_prefix="Test Set — ",
    )
    stats_utils.draw_roc_curve(
        all_results,
        saved_database_dir=cg.SAVED_DATABASE_DIR,
        unknown_slides=unknown_slides,
        amr_cutoff=None,
        title="Test Set — 2R",
    )

    # 4-class confusion at the best-accuracy threshold.
    best_key, best_nn_col = _best_threshold_column(all_results, file_dict)
    print(f"Best-accuracy threshold column: {best_key}")
    draw_four_class_confusion(
        all_results[0], best_nn_col,
        title=f"Test Set — 4-class ({best_key})",
    )


if __name__ == "__main__":
    main()
