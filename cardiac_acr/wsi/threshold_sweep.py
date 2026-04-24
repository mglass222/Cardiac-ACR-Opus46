#!/usr/bin/env python
# coding: utf-8

"""
Sweep PREDICTION_THRESHOLD over pre-thresholded patch predictions.

Reads the ``model_predictions_dict_<slide>.pickle`` files produced by
``wsi.diagnose.classify_patches`` and reports, per slide, how many
patches survive each threshold broken down by argmax class. Lets you
pick ``PREDICTION_THRESHOLD`` empirically instead of inheriting 0.99
from the legacy pipeline.

Usage:
    python -m cardiac_acr.wsi.threshold_sweep
    python -m cardiac_acr.wsi.threshold_sweep --thresholds 0.5 0.8 0.95
"""

import argparse
import os
import pickle
import re

import numpy as np

from cardiac_acr.backends.uni import config as uni_cfg
from cardiac_acr.backends.uni.evaluate import load_head_checkpoint


_PREFIX = "model_predictions_dict_"
_SLIDE_RE = re.compile(rf"^{_PREFIX}(.+?)\.pickle$")


def _discover_prediction_files(saved_database_dir):
    if not os.path.isdir(saved_database_dir):
        raise SystemExit(
            f"No prediction directory at {saved_database_dir}. "
            "Run `python -m cardiac_acr.wsi.diagnose --backend uni` first."
        )
    out = []
    for name in sorted(os.listdir(saved_database_dir)):
        if name.endswith("_filtered.pickle"):
            continue
        match = _SLIDE_RE.match(name)
        if match:
            out.append((match.group(1), os.path.join(saved_database_dir, name)))
    return out


def _sweep(predictions, num_classes, thresholds):
    """Return an int array [len(thresholds), num_classes] of surviving counts."""
    if not predictions:
        return np.zeros((len(thresholds), num_classes), dtype=int)
    probs = np.stack([np.asarray(v, dtype=np.float32) for v in predictions.values()])
    top_prob = probs.max(axis=1)
    top_class = probs.argmax(axis=1)
    counts = np.zeros((len(thresholds), num_classes), dtype=int)
    for i, t in enumerate(thresholds):
        kept_classes = top_class[top_prob > t]
        if kept_classes.size:
            counts[i] = np.bincount(kept_classes, minlength=num_classes)
    return counts


def _print_table(title, total, classes, thresholds, counts):
    print(f"\n=== {title} ({total} patches) ===")
    col_widths = [max(len(c), 6) for c in classes]
    header = (
        f"  {'threshold':<10}"
        + " ".join(f"{c:>{w}}" for c, w in zip(classes, col_widths))
        + f" {'kept':>7}"
    )
    print(header)
    for i, t in enumerate(thresholds):
        row = counts[i]
        kept = int(row.sum())
        pct = 100.0 * kept / total if total else 0.0
        cells = " ".join(f"{n:>{w}}" for n, w in zip(row, col_widths))
        print(f"  {t:<10}{cells} {kept:>7} ({pct:5.1f}%)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cardiac_acr.wsi.threshold_sweep")
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.5, 0.7, 0.9, 0.95, 0.99],
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Head checkpoint to read class ordering from.")
    parser.add_argument(
        "--saved-database-dir", default=uni_cfg.SAVED_DATABASE_DIR,
        help="Directory of model_predictions_dict_<slide>.pickle files.",
    )
    args = parser.parse_args(argv)

    _, blob = load_head_checkpoint(path=args.checkpoint)
    classes = list(blob["classes"])
    thresholds = sorted(args.thresholds)

    files = _discover_prediction_files(args.saved_database_dir)
    if not files:
        raise SystemExit(
            f"No prediction pickles under {args.saved_database_dir}. "
            "Run `python -m cardiac_acr.wsi.diagnose --backend uni` first."
        )

    aggregate = np.zeros((len(thresholds), len(classes)), dtype=int)
    aggregate_total = 0
    for slide_id, path in files:
        with open(path, "rb") as fh:
            predictions = pickle.load(fh)
        counts = _sweep(predictions, len(classes), thresholds)
        _print_table(f"slide {slide_id}", len(predictions), classes, thresholds, counts)
        aggregate += counts
        aggregate_total += len(predictions)

    _print_table(
        f"aggregate across {len(files)} slides",
        aggregate_total, classes, thresholds, aggregate,
    )


if __name__ == "__main__":
    main()
