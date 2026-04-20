#!/usr/bin/env python
# coding: utf-8

"""
Train the ResNet-50 patch classifier used by the cardiac-ACR pipeline.

Training proceeds in two phases (ported from
``Cardiac_ACR_Pytorch_V8_FINAL.ipynb``):

    1. ``train_fc_only`` — only FC + BatchNorm layers receive gradients.
    2. ``train_unlocked_layers`` — ``layer3`` and ``layer4`` are
       additionally unfrozen with reduced learning rates (lr/9 and lr/3
       respectively); the head continues to train at the full rate.

After each phase the model is written out to ``cg.MODEL_DIR`` as both a
full checkpoint and a state-dict. The fine-tuned model saved at the end
of phase 2 uses the ``_ft`` suffix and is the one loaded by
``cardiac_acr_diagnose_wsi.py`` at inference time.

Usage:
    python -m cardiac_acr.training.train
"""

import copy
import os
import time

import torch
import torch.nn as nn
import torchvision
from torch import optim

from cardiac_acr import cardiac_globals as cg
from cardiac_acr import cardiac_utils as utils
from cardiac_acr.training import data_utils
from cardiac_acr.training.model import build_resnet, unfreeze_layers


def train_model(device, model, batch_size, dataloaders, criterion, optimizer, num_epochs):
    """
    Core training loop. Alternates Training / Validation phases per epoch
    and returns the best-val-accuracy model weights.

    Returns
    -------
    model : torch.nn.Module
        Loaded with the best-validation-accuracy weights seen.
    val_acc_history : list[torch.Tensor]
        Validation accuracy at the end of each epoch.
    """
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_patches, valid_patches = data_utils.count_patches()

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

                pct_complete = 100 * (ii + 1) / num_batches
                print(
                    f"Phase: {phase} {pct_complete:.2f} % complete. "
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


def _save_model(model, model_name, suffix=""):
    """Write both the full model and its state_dict to ``cg.MODEL_DIR``."""
    utils.make_directory(cg.MODEL_DIR)
    state_dict_path = os.path.join(cg.MODEL_DIR, f"{model_name}{suffix}_state_dict")
    full_model_path = os.path.join(cg.MODEL_DIR, f"{model_name}{suffix}")

    torch.save(model.state_dict(), state_dict_path)
    torch.save(model, full_model_path)
    print(f"Model saved: {full_model_path}")


def train_fc_only(model, model_name, batch_size, dataloaders, criterion,
                  num_epochs, lr, device):
    """
    Phase 1: train only the FC head and BatchNorm layers.

    Saves the resulting weights as ``<model_name>`` in ``cg.MODEL_DIR``.
    """
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)

    print("\nTraining Only Classifier and Batch Norm Layers.\n")
    model, _ = train_model(
        device, model, batch_size, dataloaders, criterion, optimizer, num_epochs
    )

    _save_model(model, model_name)
    return model


def train_unlocked_layers(model, model_name, batch_size, dataloaders, criterion,
                          num_epochs, lr, device):
    """
    Phase 2: fine-tune ``layer3``/``layer4`` alongside the FC head.

    Uses staged learning rates — ``layer4`` at ``lr/3``, ``layer3`` at
    ``lr/9``, and everything else (FC + BN) at the full ``lr``. Saves
    the result with an ``_ft`` suffix; this is the checkpoint the
    diagnosis pipeline loads.
    """
    unfreeze_layers(model, ("layer3", "layer4"))

    base_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "layer3" not in name and "layer4" not in name
    ]

    optimizer_ft = optim.Adam([
        {"params": base_params, "lr": lr},
        {"params": model.layer3.parameters(), "lr": lr / 9},
        {"params": model.layer4.parameters(), "lr": lr / 3},
    ])

    print("\nTraining Classifier, Batch Norm Layers, Layer3, and Layer4.\n")
    model_ft, _ = train_model(
        device, model, batch_size, dataloaders, criterion, optimizer_ft, num_epochs
    )

    _save_model(model_ft, model_name, suffix="_ft")
    return model_ft


def main():
    print("PyTorch Version:", torch.__version__)
    print("Torchvision Version:", torchvision.__version__)
    print()

    model_name = cg.TRAIN_DEFAULT_MODEL_NAME
    batch_size = cg.TRAIN_BATCH_SIZE
    lr = cg.TRAIN_LEARNING_RATE
    input_size = cg.TRAIN_INPUT_SIZE
    num_epochs = cg.TRAIN_NUM_EPOCHS

    device = utils.initialize_gpu()

    dataloaders = data_utils.initialize_dataloaders(input_size, batch_size)

    classes = sorted(os.listdir(cg.TRAIN_DIR))
    print("classes =", classes)

    weights = torch.tensor(data_utils.class_weights()).to(device)
    print("class weights =", weights)

    class_percentages = data_utils.get_percentages()
    print("class_percentage in training set =", class_percentages)

    print("model =", model_name)
    print("lr =", lr)
    print("batch_size =", batch_size)
    print("nn input size =", input_size)

    criterion = nn.CrossEntropyLoss(weight=weights)

    num_classes = data_utils.count_classes()
    model = build_resnet(model_name, num_classes).to(device)

    # Phase 1: FC + BN only.
    train_fc_only(model, model_name, batch_size, dataloaders, criterion,
                  num_epochs, lr, device)

    # Phase 2: unfreeze layer3 / layer4 and fine-tune.
    train_unlocked_layers(model, model_name, batch_size, dataloaders, criterion,
                          num_epochs, lr, device)


if __name__ == "__main__":
    main()
