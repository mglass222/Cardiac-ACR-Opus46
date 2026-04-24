#!/usr/bin/env python
# coding: utf-8

"""
Backend-agnostic whole-slide diagnosis pipeline.

Accepts a :class:`cardiac_acr.backends.BackendClassifier` and runs:

  1. Preprocessing — PNG extraction, tissue filtering, tile scoring,
     tileset splitting.
  2. Classification — each patch through ``classifier.classify``, saving
     softmax probabilities to ``classifier.saved_database_dir``. The
     DataLoader workers apply the <50% tissue filter in-line, so there
     is no separate per-patch filtering pass.
  3. Threshold filtering — drop patches whose top probability is below
     ``config.PREDICTION_THRESHOLD``.
  4. Slide-level diagnosis — aggregate filtered patch labels + a
     dedicated 1R2 focus count into a single rejection grade.

Outputs land under ``classifier.saved_database_dir`` and
``classifier.slide_dx_dir``. PNG/SVS annotation writing is not yet
wired in (see DEVELOPMENT_LOG "V1 scope").

Usage:
    python -m cardiac_acr.wsi.diagnose --backend {uni,resnet}
"""

import argparse
import os
import pickle
import time
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cardiac_acr import config as cg
from cardiac_acr.backends import BackendClassifier, load_classifier
from cardiac_acr.preprocessing import filter as wsi_filter
from cardiac_acr.preprocessing.filter_patches import (
    apply_image_filters,
    tissue_percent,
)
from cardiac_acr.preprocessing import slide
from cardiac_acr.preprocessing import tiles
from cardiac_acr.preprocessing import tileset_utils
from cardiac_acr.utils import cardiac_utils as utils
from cardiac_acr.wsi import count_1r2


# Classification batch size at inference. Kept local (not on the
# BackendClassifier) because it is tuning for the diagnosis loop, not
# a property of the backend itself.
_CLASSIFY_BATCH = 64

# DataLoader workers for patch PIL decode + tissue filter + transform. 8
# saturates the 2070 SUPER during classify on this project; bump if a
# faster GPU is still CPU-bound.
_CLASSIFY_WORKERS = 8

# Minimum tissue percentage a patch must have to be classified. Matches
# the threshold used by the old disk-based filter_patches pass.
_TISSUE_PCT_MIN = 50


class _PatchFileDataset(Dataset):
    def __init__(self, patch_paths, transform):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"))
        filtered = apply_image_filters(rgb)
        if tissue_percent(filtered) < _TISSUE_PCT_MIN:
            return path, None
        tensor = self.transform(Image.fromarray(rgb))
        return path, tensor


def _drop_empty_collate(batch):
    kept = [(p, t) for p, t in batch if t is not None]
    if not kept:
        return [], None
    paths = [p for p, _ in kept]
    tensors = torch.stack([t for _, t in kept], dim=0)
    return paths, tensors


def _ensure_dirs(classifier: BackendClassifier):
    for d in (
        classifier.saved_database_dir,
        classifier.slide_dx_dir,
        cg.BACKEND_DIR,
    ):
        os.makedirs(d, exist_ok=True)


def classify_patches(slide_number, classifier: BackendClassifier,
                     batch_size=_CLASSIFY_BATCH,
                     num_workers=_CLASSIFY_WORKERS):
    """Run every patch through ``classifier.classify``, save softmax probs.

    The DataLoader workers apply the <50% tissue filter in-line and
    return a sentinel for rejects; ``_drop_empty_collate`` discards them
    before the batch reaches the GPU, so no separate filter pass is
    needed.
    """
    patch_dir = os.path.join(cg.SPLIT_TILE_DIR, str(slide_number))
    patch_names = sorted(listdir(patch_dir))
    patch_paths = [os.path.join(patch_dir, p) for p in patch_names]

    t0 = time.time()

    device = classifier.device
    dataset = _PatchFileDataset(patch_paths, classifier.transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        collate_fn=_drop_empty_collate,
    )

    predictions = {}
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"slide {slide_number} classify",
        unit="batch",
        dynamic_ncols=True,
        mininterval=0.5,
        position=1,
        leave=False,
    )
    for batch_paths, batch in progress:
        if batch is None:
            continue
        batch = batch.to(device, non_blocking=True)
        logits = classifier.classify(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for path, prob in zip(batch_paths, probs):
            predictions[path] = prob

    out_path = os.path.join(
        classifier.saved_database_dir,
        f"model_predictions_dict_{slide_number}.pickle",
    )
    with open(out_path, "wb") as fh:
        pickle.dump(predictions, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  kept {len(predictions)}/{len(patch_paths)} tissue patches "
          f"({time.time() - t0:.1f}s) -> {out_path}")


def threshold_predictions(slide_number, classifier: BackendClassifier,
                          threshold=None):
    """Drop any patch whose top probability is below ``threshold``."""
    if threshold is None:
        threshold = cg.PREDICTION_THRESHOLD

    in_path = os.path.join(
        classifier.saved_database_dir,
        f"model_predictions_dict_{slide_number}.pickle",
    )
    with open(in_path, "rb") as fh:
        predictions = pickle.load(fh)

    filtered = {
        k: v for k, v in predictions.items()
        if np.asarray(v).max() > threshold
    }
    print(f"  thresholded: kept {len(filtered)}/{len(predictions)} patches")

    out_path = os.path.join(
        classifier.saved_database_dir,
        f"model_predictions_dict_{slide_number}_filtered.pickle",
    )
    with open(out_path, "wb") as fh:
        pickle.dump(filtered, fh, protocol=pickle.HIGHEST_PROTOCOL)


def diagnose(slide_number, classifier: BackendClassifier):
    """Aggregate patch-level predictions into a slide-level rejection grade."""
    in_path = os.path.join(
        classifier.saved_database_dir,
        f"model_predictions_dict_{slide_number}_filtered.pickle",
    )
    with open(in_path, "rb") as fh:
        filtered_predictions = pickle.load(fh)

    classes = classifier.classes
    class_count = {name: 0 for name in classes}
    for value in filtered_predictions.values():
        idx = int(np.argmax(value))
        class_count[classes[idx]] += 1

    # 1R2 focus count comes from the dedicated segmentation pipeline,
    # not from the patch classifier.
    one_r_two_count = count_1r2.main(slide_number, classifier.saved_database_dir)
    class_count["1R2"] = one_r_two_count
    print(f"  class counts: {class_count}")

    one_r_one_a = class_count.get("1R1A", 0)
    if one_r_one_a == 0 and one_r_two_count == 0:
        dx = "0R"
    elif one_r_one_a > 0 and one_r_two_count == 0:
        dx = "1R1A"
    elif one_r_two_count > 0:
        dx = "1R2" if one_r_two_count < 2 else "2R"
    else:
        dx = "UNK"

    print(f"  slide {slide_number} dx = {dx}")

    dx_pickle = os.path.join(
        classifier.slide_dx_dir,
        f"slide_dx_dict_{int(cg.PREDICTION_THRESHOLD * 100)}_pct.pickle",
    )
    if os.path.exists(dx_pickle):
        with open(dx_pickle, "rb") as fh:
            slide_dx_dict = pickle.load(fh)
    else:
        slide_dx_dict = {}
    slide_dx_dict[slide_number] = dx
    with open(dx_pickle, "wb") as fh:
        pickle.dump(slide_dx_dict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return dx, class_count


def run(backend: str, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    classifier = load_classifier(backend, device=device,
                                 checkpoint_path=checkpoint_path)
    print(f"backend: {classifier.name} ({len(classifier.classes)} classes)")
    _ensure_dirs(classifier)

    slides_to_process = utils.get_test_slide_numbers()
    print("slides to process:", slides_to_process)

    t0 = time.time()
    slide.multiprocess_training_slides_to_images(image_num_list=slides_to_process)
    wsi_filter.multiprocess_apply_filters_to_images(image_num_list=slides_to_process)
    tiles.multiprocess_filtered_images_to_tiles(
        save_top_tiles=False, image_num_list=slides_to_process
    )
    for s in slides_to_process:
        tileset_utils.process_tilesets_multiprocess(s)
    print(f"preprocessing done in {time.time() - t0:.1f}s")

    slide_bar = tqdm(
        slides_to_process,
        desc="slides",
        unit="slide",
        dynamic_ncols=True,
        position=0,
    )
    for slide_number in slide_bar:
        slide_bar.set_description(f"slides (current: {slide_number})")
        slide_t0 = time.time()

        classify_patches(slide_number, classifier)
        threshold_predictions(slide_number, classifier)
        diagnose(slide_number, classifier)

        tqdm.write(f"slide {slide_number} done in {time.time() - slide_t0:.1f}s")

    slide_bar.close()
    print(f"\nTotal time: {time.time() - t0:.1f}s")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cardiac_acr.wsi.diagnose",
        description="Run end-to-end WSI diagnosis with the chosen backend.",
    )
    parser.add_argument("--backend", choices=("uni", "resnet"), required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Path to the backend checkpoint (defaults to the "
                             "backend's MODEL_DIR).")
    args = parser.parse_args(argv)
    run(args.backend, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
