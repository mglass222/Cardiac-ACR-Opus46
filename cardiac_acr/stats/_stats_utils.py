#!/usr/bin/env python
# coding: utf-8

"""
Shared helpers for the slide-level stats scripts.

``training_set_stats.py`` and ``test_set_stats.py`` both consume a
directory of per-threshold diagnosis CSVs plus a pathologist
ground-truth CSV, and produce the same artifacts — a combined summary
CSV, a grid of confusion matrices across thresholds, and an ROC curve
derived from summed 1R2 probabilities per slide.

Both notebooks originally duplicated almost every helper below. They
now import from this module.
"""

import csv
import math
import os
import pickle
from os import listdir

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from cardiac_acr import cardiac_globals as cg


# Every diagnosis label that the neural network / pathologist can emit.
ALL_DX_CLASSES = [
    "0R", "1R1A", "1R1B", "1R2", "1R",
    "Healing", "Hemorrhage", "Normal", "Quilty",
    "2R", "2R3A", "3R",
]

# Labels considered "no rejection" (or mild-only) for the binary view.
_NON_REJECTION_LABELS = {"0R", "1R1A", "1R1B", "1R", "1R2"}
_REJECTION_LABELS = {"2R", "2R3A", "3R"}


def convert_to_binary(dx_list):
    """
    Map a list of diagnosis strings to ``{0, 1}`` for 2R vs. not-2R.

    Unknown labels are logged and skipped — the return list may be
    shorter than the input.
    """
    bin_dx = []
    for item in dx_list:
        item = item.strip()
        if item in _NON_REJECTION_LABELS:
            bin_dx.append(0)
        elif item in _REJECTION_LABELS:
            bin_dx.append(1)
        else:
            print(f"Error, entry not found: {item!r}")
    return bin_dx


def convert_to_binary_for_class(dx_list, pos_class):
    """
    One-vs-rest binarization: ``pos_class`` → 1, everything else → 0.
    """
    bin_dx = []
    neg_classes = [c for c in ALL_DX_CLASSES if c != pos_class]
    for item in dx_list:
        item = item.strip()
        if item in neg_classes:
            bin_dx.append(0)
        elif item == pos_class:
            bin_dx.append(1)
        else:
            print(f"Error, entry not found: {item!r}")
    return bin_dx


def get_dx_files(results_path):
    """
    Discover per-threshold NN-diagnosis CSVs inside ``results_path``.

    Returns a ``dict`` mapping a threshold-identifier string (derived
    from the filename) to the absolute file path.
    """
    file_dict = {}
    files = [f for f in listdir(results_path) if f.lower().endswith(".csv")]
    for file in files:
        # Preserve the original notebook's identifier extraction rule:
        # use everything after the third underscore in the stem.
        stem = file.rsplit(".", 1)[0]
        combo = "_".join(stem.split("_")[3:])
        file_dict[combo] = os.path.join(results_path, file)
    return file_dict


def make_summary_csv(summary_csv_path, ground_truth_csv, file_dict):
    """
    Initialize ``summary_csv_path`` with pathologist dx + empty columns
    for every threshold combination discovered by :func:`get_dx_files`.

    Returns ``(all_results, unknown_slides, slide_list)``.

    ``all_results[0]`` is the pathologist-dx column. Subsequent entries
    are appended by :func:`add_results_to_csv`.
    """
    percent_list = list(file_dict.keys())

    all_results = []
    unknown_slides = []
    slide_list = []
    path_dx_column = []

    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)

    with open(summary_csv_path, "w", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["Slide", "Path_Dx"] + percent_list)

        with open(ground_truth_csv, "r", newline="") as in_csv:
            reader = csv.reader(in_csv)
            next(reader)  # skip header
            for row in reader:
                writer.writerow(row)
                slide_list.append(row[0])
                path_dx_column.append(row[1])
                if row[1] == "":
                    unknown_slides.append(row[0].zfill(3))

    all_results.append(path_dx_column)
    return all_results, unknown_slides, slide_list


def add_results_to_csv(summary_csv_path, file_dict, all_results):
    """
    Append each per-threshold diagnosis column into ``summary_csv_path``.

    Mutates and returns ``all_results``.
    """
    # Read back the existing contents so we can widen each row.
    with open(summary_csv_path, "r", newline="") as csvfile:
        sheet_contents = list(csv.reader(csvfile))

    for result_file in file_dict.values():
        file_results = []
        with open(result_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header
            for row in reader:
                file_results.append(row[1])

        for i, value in enumerate(file_results):
            sheet_contents[i + 1].append(value)

        all_results.append(file_results)

    with open(summary_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in sheet_contents:
            writer.writerow(row)

    return all_results


def filter_csv(all_results, slide_list, amr_cutoff=None):
    """
    Drop rows with unknown pathologist dx (and, optionally, AMR slides).

    Parameters
    ----------
    all_results : list[list[str]]
        Output of :func:`make_summary_csv` + :func:`add_results_to_csv`.
        ``all_results[0]`` is the path-dx column.
    slide_list : list[str]
        Slide numbers as strings, in the same order as ``all_results``.
    amr_cutoff : int or None
        If given, drop slides whose numeric id is ``>= amr_cutoff``.
        The training-set notebook used ``279`` to exclude AMR slides.

    Returns
    -------
    filtered_results : list[list[str]]
        Same shape as ``all_results`` with filtered rows removed.
    remove_idxs : list[int]
        Indices (in the original order) that were removed.
    """
    path_dx = all_results[0]
    unknown_idxs = [i for i, dx in enumerate(path_dx) if dx == ""]

    if amr_cutoff is not None:
        amr_idxs = [
            i for i, slide in enumerate(slide_list)
            if slide.isdigit() and int(slide) >= amr_cutoff
        ]
    else:
        amr_idxs = []

    remove_idxs = sorted(set(unknown_idxs) | set(amr_idxs), reverse=True)

    filtered_results = []
    for column in all_results:
        col = list(column)
        for idx in remove_idxs:
            del col[idx]
        filtered_results.append(col)

    return filtered_results, remove_idxs


def draw_confusion_mtx(all_results, file_dict, title_prefix=""):
    """
    Plot a grid of binary (2R vs not-2R) confusion matrices, one per
    threshold column in ``all_results[1:]``.

    The best-accuracy tile is drawn with a ``'rainbow'`` colormap so it
    stands out. Returns the F1 score of the best threshold.
    """
    percent_list = list(file_dict.keys())

    all_results_bin = [convert_to_binary(col) for col in all_results]
    path_dx_bin = all_results_bin[0]
    nn_results_bin = all_results_bin[1:]

    num_confmtx = len(nn_results_bin)
    num_cols = 4
    num_rows = max(1, math.ceil(num_confmtx / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.set_size_inches(20, 10)
    # Normalize axs to 2D for uniform indexing.
    if num_rows == 1:
        axs = np.array([axs])

    confusion_matrices = []
    accuracies = []
    for nn_col in nn_results_bin:
        confusion_matrices.append(confusion_matrix(path_dx_bin, nn_col))
        accuracies.append(round(accuracy_score(path_dx_bin, nn_col), 4))

    best_acc_idx = accuracies.index(max(accuracies))

    count = 0
    for x in range(num_rows):
        for y in range(num_cols):
            if count >= len(confusion_matrices):
                axs[x, y].axis("off")
                continue

            cmap = "rainbow" if count == best_acc_idx else None
            axs[x, y].matshow(confusion_matrices[count], cmap=cmap)

            for (i, j), z in np.ndenumerate(confusion_matrices[count]):
                axs[x, y].text(j, i, f"{z:0.1f}", ha="center", va="center")

            threshold_label = percent_list[count].split("_")[0] + "%"
            axs[x, y].set_title(
                f"{title_prefix}Threshold {threshold_label} : "
                f"Accuracy {accuracies[count]}"
            )
            count += 1

    best_f1 = f1_score(path_dx_bin, nn_results_bin[best_acc_idx])
    print(f"Best F1_Score = {best_f1}")
    return best_f1


def draw_roc_curve(all_results, saved_database_dir=None,
                   unknown_slides=None, amr_cutoff=None, title="2R"):
    """
    ROC curve for binary rejection classification using the sum of
    per-patch 1R2 probabilities on each slide as the score.

    Parameters
    ----------
    all_results : list[list[str]]
        Filtered results; ``all_results[0]`` is the binary ground truth.
    saved_database_dir : str or None
        Directory of ``model_predictions_dict_*.pickle`` files. Defaults
        to ``cg.SAVED_DATABASE_DIR``.
    unknown_slides : set[str] or None
        Slide numbers (zero-padded strings) to exclude.
    amr_cutoff : int or None
        If given, also exclude slides whose numeric id is ``>= amr_cutoff``.
    """
    if saved_database_dir is None:
        saved_database_dir = cg.SAVED_DATABASE_DIR
    unknown_slides = set(unknown_slides or [])

    all_results_bin = [convert_to_binary(col) for col in all_results]
    path_dx_bin = all_results_bin[0]

    def _keep(filename):
        if "filtered" in filename:
            return False
        # Expected pattern: model_predictions_dict_<slide>.pickle
        slide_num = filename.rsplit(".", 1)[0].split("_")[3]
        if slide_num in unknown_slides:
            return False
        if amr_cutoff is not None and slide_num.isdigit() and int(slide_num) >= amr_cutoff:
            return False
        return True

    prediction_dicts = [f for f in listdir(saved_database_dir) if _keep(f)]

    slide_1r2_counts = []
    for filename in prediction_dicts:
        with open(os.path.join(saved_database_dir, filename), "rb") as handle:
            model_predictions_dict = pickle.load(handle)
        _1r2_sum = sum(probs[cg.CLASS_TO_INDEX["1R2"]]
                       for probs in model_predictions_dict.values())
        slide_1r2_counts.append(_1r2_sum)

    roc_auc = roc_auc_score(path_dx_bin, slide_1r2_counts)
    fpr, tpr, _ = roc_curve(path_dx_bin, slide_1r2_counts)
    print(f"ROC AUC = {roc_auc}")

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(title, size=15)
    plt.plot(fpr, tpr, color="blue", linewidth=3,
             label="ROC curve (area = %0.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([-0.01, 1, 0, 1.01])
    plt.xlabel("FPR", size=15)
    plt.ylabel("TPR", size=15)
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc
