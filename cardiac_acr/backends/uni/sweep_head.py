#!/usr/bin/env python
# coding: utf-8

"""
Hyperparameter sweep for the UNI classifier head.

Trains the head once per (head_type, lr, weight_decay) combination on
the current feature cache and prints a ranked summary of best
validation accuracies. Does not save checkpoints — the main
``train.py`` entry point is the way to commit a run.

Usage:
    python -m cardiac_acr.backends.uni.sweep_head
"""

import csv
import itertools
import os
import time

import torch

from cardiac_acr.backends.uni import config as uni_cfg
from cardiac_acr.backends.uni.train import train_head


# Knobs to sweep. Keep this small — 18 configs * ~35 s = ~10 min.
HEAD_TYPES = ("linear", "mlp")
LEARNING_RATES = (1e-3, 3e-4, 1e-4)
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"training cache: {uni_cfg.TRAINING_FEATURES_PATH}")
    print(f"validation cache: {uni_cfg.VALIDATION_FEATURES_PATH}")

    configs = list(itertools.product(HEAD_TYPES, LEARNING_RATES, WEIGHT_DECAYS))
    print(f"\nrunning {len(configs)} configurations\n")

    results = []
    t0 = time.time()
    for i, (head_type, lr, wd) in enumerate(configs, 1):
        run_t0 = time.time()
        _, best_acc = train_head(
            head_type=head_type,
            lr=lr,
            weight_decay=wd,
            device=device,
            save=False,
            verbose=False,
        )
        run_elapsed = time.time() - run_t0
        results.append({
            "head_type": head_type,
            "lr": lr,
            "weight_decay": wd,
            "best_val_acc": best_acc,
            "runtime_s": run_elapsed,
        })
        print(
            f"[{i:>2}/{len(configs)}] "
            f"head={head_type:<6} lr={lr:.0e} wd={wd:.0e} | "
            f"best val acc {best_acc:.4f} | {run_elapsed:.1f}s"
        )

    total_elapsed = time.time() - t0
    print(f"\nSwept {len(configs)} configs in {total_elapsed:.1f}s")

    # Ranked summary, best first.
    print("\n=== Ranked by best val acc ===")
    ranked = sorted(results, key=lambda r: -r["best_val_acc"])
    print(f"{'rank':<5}{'head':<8}{'lr':<10}{'wd':<10}{'best val acc':<14}")
    for rank, r in enumerate(ranked, 1):
        print(
            f"{rank:<5}{r['head_type']:<8}{r['lr']:<10.0e}"
            f"{r['weight_decay']:<10.0e}{r['best_val_acc']:<14.4f}"
        )

    # Persist a CSV next to the feature cache for the dev log.
    log_dir = os.path.join(uni_cfg.DATA_DIR, "Logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "head_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
