#!/usr/bin/env python
# coding: utf-8

"""
5-fold cross-validation runner for the cardiac-ACR patch classifier.

For each of 5 folds:
    1. Rebuild ``CROSS_VAL_TRAIN_DIR`` / ``CROSS_VAL_VALID_DIR`` from the
       master patch library at ``cg.OPENSLIDE_DIR`` — each class is split
       into 5 contiguous chunks and one chunk is held out as validation.
    2. Train a fresh ResNet-50 (FC + BN only) for ``num_epochs`` epochs.
    3. Save the trained model under ``cg.CROSS_VAL_MODEL_DIR``.
    4. Run the trained model over this fold's validation patches and
       append the (softmax-probabilities, ground-truth-label) pairs
       into ``model_predictions_dict.pickle`` under ``cg.CROSS_VAL_DIR``.

At the end, a ``crossvalidation_results.pickle`` with per-fold best
validation accuracy is written to the same directory.

Ported from ``Cardiac_ACR_Pytorch_V8_CrossValidation.ipynb``. The
exploratory plotting cells from the bottom of the original notebook
have been dropped — equivalent figures are produced by the scripts
under ``Code/stats/``.

NOTE: This script currently duplicates the training loop from
``train.py``. A future refactor should have ``cross_validation.py``
call into shared training helpers rather than inlining them.
"""

import copy
import os
import pickle
import shutil
import sys
import time
from os import listdir
from os.path import isdir

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch import optim
from torchvision import transforms

# Allow imports from the parent ``Code/`` directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cardiac_globals as cg  # noqa: E402
import cardiac_utils as utils  # noqa: E402

from training import data_utils  # noqa: E402
from training.model import build_resnet  # noqa: E402


NUM_FOLDS = 5
PREDICTIONS_PICKLE = "model_predictions_dict.pickle"
RESULTS_PICKLE = "crossvalidation_results.pickle"


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------


def _delete_files(path):
    """Remove every file directly inside ``path`` (non-recursive)."""
    if not os.path.isdir(path):
        return
    for file in listdir(path):
        full = os.path.join(path, file)
        if os.path.isfile(full):
            os.remove(full)


def create_training_sets(fold_index):
    """
    Populate ``CROSS_VAL_TRAIN_DIR`` / ``CROSS_VAL_VALID_DIR`` for one fold.

    The master patch library at ``cg.OPENSLIDE_DIR`` is partitioned into
    5 contiguous chunks per class. ``fold_index`` (0..4) selects which
    chunk becomes the validation set; the remaining four are training.
    """
    classes = listdir(cg.OPENSLIDE_DIR)

    utils.make_directory(cg.CROSS_VAL_TRAIN_DIR)
    utils.make_directory(cg.CROSS_VAL_VALID_DIR)

    for class_name in classes:
        src_dir = os.path.join(cg.OPENSLIDE_DIR, class_name)
        if not isdir(src_dir):
            continue

        src_files = sorted(listdir(src_dir))
        num_files = len(src_files)
        cutoff = num_files // NUM_FOLDS

        if fold_index < NUM_FOLDS - 1:
            val_files = src_files[cutoff * fold_index : cutoff * (fold_index + 1)]
        else:
            val_files = src_files[cutoff * fold_index :]

        val_set = set(val_files)
        train_files = [f for f in src_files if f not in val_set]

        print(f"Getting Training Files for: {class_name}")
        train_dst = os.path.join(cg.CROSS_VAL_TRAIN_DIR, class_name)
        utils.make_directory(train_dst)
        _delete_files(train_dst)
        for file in train_files:
            shutil.copy(os.path.join(src_dir, file), train_dst)

        print(f"Getting Validation Files for: {class_name}")
        val_dst = os.path.join(cg.CROSS_VAL_VALID_DIR, class_name)
        utils.make_directory(val_dst)
        _delete_files(val_dst)
        for file in val_files:
            shutil.copy(os.path.join(src_dir, file), val_dst)

    print("\nDONE\n")


# ---------------------------------------------------------------------------
# Training loop (fold-scoped copy of train.py, operating on CROSS_VAL_* dirs)
# ---------------------------------------------------------------------------


def _fold_dataloaders(input_size, batch_size):
    """
    Dataloaders for the current fold using ``CROSS_VAL_*`` directories.

    Mirrors :func:`data_utils.initialize_dataloaders` but points at the
    cross-validation split rather than the main training split.
    """
    return data_utils.initialize_dataloaders(
        input_size=input_size,
        batch_size=batch_size,
        training_root=os.path.dirname(cg.CROSS_VAL_TRAIN_DIR),
    )


def _train_fold(device, model, batch_size, dataloaders, criterion, optimizer, num_epochs):
    """Fold training loop. Returns (best-weights model, val acc history)."""
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_patches, valid_patches = data_utils.count_patches(
        train_dir=cg.CROSS_VAL_TRAIN_DIR, valid_dir=cg.CROSS_VAL_VALID_DIR
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        timer = time.time()

        for phase in ("Training", "Validation"):
            if phase == "Training":
                model.train()
                num_batches = train_patches / batch_size
            else:
                model.eval()
                num_batches = valid_patches / batch_size

            running_loss = 0.0
            running_corrects = 0

            for ii, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "Training"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "Training":
                        loss.backward()
                        optimizer.step()

                pct = 100 * (ii + 1) / num_batches
                print(
                    f"Phase: {phase} {pct:.2f} % complete. "
                    f"{time.time() - timer:.2f} seconds elapsed in epoch.",
                    end="\r",
                )

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "Validation":
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def _train_fc_only_fold(model, model_name, batch_size, dataloaders, criterion,
                        num_epochs, lr, device, fold_index):
    """Phase-1 training + checkpoint save for a single fold."""
    print("Training ResNet: Only Training Classifier and Batch Norm Layers.")

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)

    model, hist = _train_fold(
        device, model, batch_size, dataloaders, criterion, optimizer, num_epochs
    )

    utils.make_directory(cg.CROSS_VAL_MODEL_DIR)
    state_dict_path = os.path.join(
        cg.CROSS_VAL_MODEL_DIR, f"{model_name}_{fold_index}_state_dict"
    )
    full_path = os.path.join(cg.CROSS_VAL_MODEL_DIR, f"{model_name}_{fold_index}")
    torch.save(model.state_dict(), state_dict_path)
    torch.save(model, full_path)
    print(f"Model saved: {full_path}")

    return model, hist


# ---------------------------------------------------------------------------
# Per-fold validation-set prediction dump
# ---------------------------------------------------------------------------


def _model_predict_batch(batch, model, input_size, device):
    """Run a list of PIL images through the model and return raw logits."""
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(cg.IMAGENET_MEAN, cg.IMAGENET_STD),
    ])
    batch_tensor = torch.stack([transform(img) for img in batch]).to(device)
    return model(batch_tensor)


def _get_label(folder_name):
    """Map a class folder name to its integer class index."""
    return cg.CLASS_TO_INDEX.get(folder_name)


def _model_predict_fold(model, fold_index, input_size, batch_size, device):
    """
    Classify every patch in the fold's validation directory.

    Appends into (or creates) ``CROSS_VAL_DIR/model_predictions_dict.pickle``
    keyed by patch path, with values ``(softmax_probs, ground_truth_label)``.
    """
    t0 = time.time()
    print("Generating predictions on this fold's validation patches.")

    pickle_path = os.path.join(cg.CROSS_VAL_DIR, PREDICTIONS_PICKLE)
    if fold_index == 0 or not os.path.exists(pickle_path):
        model_predictions_dict = {}
    else:
        with open(pickle_path, "rb") as handle:
            model_predictions_dict = pickle.load(handle)

    folder_names = listdir(cg.CROSS_VAL_VALID_DIR)

    for folder_name in folder_names:
        folder = os.path.join(cg.CROSS_VAL_VALID_DIR, folder_name)
        patches = [os.path.join(folder, p) for p in listdir(folder)]
        label = _get_label(folder_name)

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

            preds = _model_predict_batch(batch, model, input_size, device)
            preds = torch.nn.functional.softmax(preds, dim=1).cpu().detach().numpy()

            for j, patch_path in enumerate(batch_patches):
                if patch_path in model_predictions_dict:
                    print("duplicate entry found")
                else:
                    model_predictions_dict[patch_path] = (preds[j], label)

        with open(pickle_path, "wb") as handle:
            pickle.dump(model_predictions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done with predictions. this took {time.time() - t0:.1f} seconds")


def _print_model_predictions():
    """Sanity-check dump of the fold predictions pickle."""
    import random

    pickle_path = os.path.join(cg.CROSS_VAL_DIR, PREDICTIONS_PICKLE)
    with open(pickle_path, "rb") as handle:
        model_predictions_dict = pickle.load(handle)

    print("Number of predictions in model prediction dict =", len(model_predictions_dict))
    print("Example prediction...")
    print(random.choice(list(model_predictions_dict.items())))


def _save_crossval_results(val_results):
    """Write per-fold best-val-accuracy results to ``cg.CROSS_VAL_DIR``."""
    pickle_path = os.path.join(cg.CROSS_VAL_DIR, RESULTS_PICKLE)
    with open(pickle_path, "wb") as handle:
        pickle.dump(val_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print("PyTorch Version:", torch.__version__)
    print("Torchvision Version:", torchvision.__version__)
    print()

    val_results = {}

    model_name = "resnet50"
    batch_size = 200
    input_size = cg.TRAIN_INPUT_SIZE
    lr = cg.TRAIN_LEARNING_RATE
    num_epochs = 10

    device = utils.initialize_gpu()

    # Class weights come from the master patch library — independent of fold.
    weights = torch.tensor(
        data_utils.class_weights(train_dir=cg.CROSS_VAL_TRAIN_DIR
                                 if os.path.isdir(cg.CROSS_VAL_TRAIN_DIR)
                                 else cg.OPENSLIDE_DIR)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    for fold_index in range(NUM_FOLDS):
        print(f"\nSTARTING CROSSVALIDATION STEP # {fold_index + 1}\n")

        create_training_sets(fold_index)

        # Recompute class weights for this fold now that the split exists.
        fold_weights = torch.tensor(
            data_utils.class_weights(train_dir=cg.CROSS_VAL_TRAIN_DIR)
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=fold_weights)

        num_classes = data_utils.count_classes(train_dir=cg.CROSS_VAL_TRAIN_DIR)
        model = build_resnet(model_name, num_classes).to(device)

        dataloaders = _fold_dataloaders(input_size, batch_size)

        model, val_acc_hx = _train_fc_only_fold(
            model, model_name, batch_size, dataloaders, criterion,
            num_epochs, lr, device, fold_index,
        )

        best_val_acc = max(val_acc_hx).cpu().detach().numpy()
        val_results[fold_index] = best_val_acc

        _model_predict_fold(model, fold_index, input_size, batch_size, device)
        _print_model_predictions()

    _save_crossval_results(val_results)

    for k, v in val_results.items():
        print(k, v)

    # Summary statistics across folds.
    fold_accs = np.array(list(val_results.values()), dtype=float)
    print(f"\nMean best val accuracy: {fold_accs.mean():.4f}")
    print(f"Std best val accuracy : {fold_accs.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
