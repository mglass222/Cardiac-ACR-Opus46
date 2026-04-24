#!/usr/bin/env python
# coding: utf-8

"""
UNI2-h backend configuration.

Re-exports the shared top-level config so modules in this subpackage can
keep a single ``from cardiac_acr.backends.uni import config as uni_cfg``
import and reach both shared paths (``TRAIN_DIR``, ``IMAGENET_MEAN`` …)
and UNI-specific hyperparameters (``UNI_MODEL_ID``, ``EMBED_DIM``,
head-training rates …) plus the per-backend output locations below.
"""

import os

from cardiac_acr.config import *  # noqa: F401,F403  (re-export shared constants)
from cardiac_acr.config import BACKEND_DIR, DATA_DIR, WSI_DIR


#####################################################################
# UNI-specific output locations
#####################################################################

FEATURE_DIR = os.path.join(DATA_DIR, "Features")
MODEL_DIR = os.path.join(DATA_DIR, "Saved_Models", "UNI_Head")

# Per-backend WSI diagnosis outputs. The UNI backend owns the
# unsuffixed names; the ResNet backend nests under a Weighted_Loss/
# subfolder in its own config so the two don't collide.
SAVED_DATABASE_DIR = os.path.join(BACKEND_DIR, "Saved_Databases")
SLIDE_DX_DIR = os.path.join(BACKEND_DIR, "Slide_Dx")
ANNOTATED_PNG_DIR = os.path.join(BACKEND_DIR, "Annotated_Test_Slides")
TEST_SLIDE_PREDICTIONS_DIR = os.path.join(BACKEND_DIR, "Test_Slide_Predictions")
TEST_SLIDE_ANNOTATIONS_DIR = os.path.join(WSI_DIR, "TEST_SLIDE_ANNOTATIONS")

# One cached tensor per split. Each file stores
# {"features": FloatTensor [N, EMBED_DIM], "labels": LongTensor [N], "classes": list[str]}.
TRAINING_FEATURES_PATH = os.path.join(FEATURE_DIR, "training.pt")
VALIDATION_FEATURES_PATH = os.path.join(FEATURE_DIR, "validation.pt")


#####################################################################
# UNI backbone
#####################################################################

UNI_MODEL_ID = "hf-hub:MahmoodLab/UNI2-h"
EMBED_DIM = 1536                 # UNI2-h CLS-token dimension
INPUT_SIZE = 224                 # UNI2-h expects 224x224


#####################################################################
# UNI feature encoding
#####################################################################

ENCODE_BATCH_SIZE = 32           # fits in ~5 GB on 8 GB cards in fp16/bf16; drop if OOM
ENCODE_NUM_WORKERS = 8


#####################################################################
# Classifier head
#####################################################################

HEAD_TYPE = "mlp"                # "linear" | "mlp"
HEAD_DROPOUT = 0.4               # only used when HEAD_TYPE == "mlp"
HEAD_HIDDEN_DIM = 512            # only used when HEAD_TYPE == "mlp"


#####################################################################
# Head training
#####################################################################

TRAIN_BATCH_SIZE = 512           # features are tiny — large batches are fine
TRAIN_LEARNING_RATE = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_NUM_EPOCHS = 50
TRAIN_COSINE_WARMUP_EPOCHS = 2
