#!/usr/bin/env python
# coding: utf-8

"""
Run the trained ``resnet50_ft`` model over every patch in the training
set and persist the resulting patch-level predictions.

Output: ``cg.TRAIN_SET_PREDICTIONS_PICKLE`` — a list of
``[ground_truth_label, softmax_probabilities]`` pairs. This pickle is
consumed by :mod:`patch_level_stats` to build the 6-class training-set
confusion matrix and the per-class ROC curves.

Ported from the lower half of ``Cardiac_ACR_Pytorch_Training_Set_Stats_V6.ipynb``.
"""

import os
import pickle
import time
from os import listdir

import torch
from PIL import Image
from torchvision import transforms

from cardiac_acr import cardiac_globals as cg
from cardiac_acr import cardiac_utils as utils


def _build_transform(input_size):
    """Standard ImageNet-normalized inference transform."""
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(cg.IMAGENET_MEAN, cg.IMAGENET_STD),
    ])


def _predict_batch(batch, model, transform, device):
    """Run a list of PIL images through the model; return raw logits."""
    batch_tensor = torch.stack([transform(img) for img in batch]).to(device)
    return model(batch_tensor)


def predict_all_patches(model, input_size, batch_size, device,
                        patch_root=None):
    """
    Classify every patch under ``patch_root`` (defaults to
    ``cg.OPENSLIDE_DIR``) and return a list of ``[label, probs]`` pairs.
    """
    if patch_root is None:
        patch_root = cg.OPENSLIDE_DIR

    transform = _build_transform(input_size)
    model_predictions = []

    t0 = time.time()
    print("Generating predictions on all patches in training set")

    for folder_name in listdir(patch_root):
        folder = os.path.join(patch_root, folder_name)
        if not os.path.isdir(folder):
            continue

        patches = [os.path.join(folder, p) for p in listdir(folder)]
        label = cg.CLASS_TO_INDEX.get(folder_name)
        if label is None:
            print(f"Skipping unknown class folder: {folder_name!r}")
            continue

        num_patches = len(patches)
        num_batches = num_patches // batch_size
        last_batch = num_patches - batch_size * num_batches

        print()
        print("Label =", label)
        print("Number of patches in the directory =", num_patches)
        print("Number of batches =", num_batches)
        print(f"Preparing to classify {num_patches} patches")

        count = 0
        for i in range(num_batches + 1):
            batch_size_i = batch_size
            if i == num_batches:
                if last_batch == 0:
                    break
                batch_size_i = last_batch

            batch = []
            batch_patches = []
            for _ in range(batch_size_i):
                patch = patches[count]
                batch_patches.append(patch)
                batch.append(Image.open(patch))
                count += 1

            if i % 10 == 0:
                print(f"Processed {i} out of {num_batches} batches")

            preds = _predict_batch(batch, model, transform, device)
            preds = torch.nn.functional.softmax(preds, dim=1).cpu().detach().numpy()

            for j in range(len(batch_patches)):
                model_predictions.append([label, preds[j]])

    print(f"\nDone in {time.time() - t0:.1f} seconds")
    return model_predictions


def main():
    device = utils.initialize_gpu()

    model_path = os.path.join(cg.MODEL_DIR, "resnet50_ft")
    print(f"Loading model from {model_path}")
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    predictions = predict_all_patches(
        model,
        input_size=cg.TRAIN_INPUT_SIZE,
        batch_size=cg.BATCH_SIZE,
        device=device,
    )

    out_path = cg.TRAIN_SET_PREDICTIONS_PICKLE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(predictions)} patch predictions to {out_path}")


if __name__ == "__main__":
    main()
