#!/usr/bin/env python
# coding: utf-8

"""
Slide-level statistics for the training set.

For each confidence-threshold sweep performed by the slide-classifier,
this script:

1. Merges the per-threshold NN diagnosis CSVs under
   ``cg.TRAIN_SET_ANALYSIS_DIR`` with the pathologist ground-truth CSV
   at ``cg.TRAIN_DX_CSV`` into a single summary at
   ``cg.TRAIN_SUMMARY_CSV``.
2. Filters out slides with unknown pathologist diagnosis and AMR slides
   (numeric id ``>= 279``).
3. Plots a grid of binary (2R vs not-2R) confusion matrices, one per
   threshold.
4. Plots a single 2R-vs-not-2R ROC curve using the sum of per-patch
   ``1R2`` probabilities on each slide as the score.

Ported from the first half of ``Cardiac_ACR_Pytorch_Training_Set_Stats_V6.ipynb``
(cells 1-10). All duplicated helpers live in :mod:`_stats_utils`.
"""

from cardiac_acr import cardiac_globals as cg
from cardiac_acr.stats import _stats_utils as stats_utils


# Training set excludes AMR slides (numeric id >= 279) from both
# threshold-grid and ROC analyses.
AMR_CUTOFF = 279


def main():
    file_dict = stats_utils.get_dx_files(cg.TRAIN_SET_ANALYSIS_DIR)
    if not file_dict:
        raise FileNotFoundError(
            f"No per-threshold diagnosis CSVs found in "
            f"{cg.TRAIN_SET_ANALYSIS_DIR}"
        )

    all_results, unknown_slides, slide_list = stats_utils.make_summary_csv(
        cg.TRAIN_SUMMARY_CSV, cg.TRAIN_DX_CSV, file_dict,
    )
    all_results = stats_utils.add_results_to_csv(
        cg.TRAIN_SUMMARY_CSV, file_dict, all_results,
    )
    all_results, _ = stats_utils.filter_csv(
        all_results, slide_list, amr_cutoff=AMR_CUTOFF,
    )

    stats_utils.draw_confusion_mtx(
        all_results, file_dict, title_prefix="Training Set — ",
    )
    stats_utils.draw_roc_curve(
        all_results,
        saved_database_dir=cg.SAVED_DATABASE_DIR,
        unknown_slides=unknown_slides,
        amr_cutoff=AMR_CUTOFF,
        title="Training Set — 2R",
    )


if __name__ == "__main__":
    main()
