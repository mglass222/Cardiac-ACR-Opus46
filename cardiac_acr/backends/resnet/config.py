#!/usr/bin/env python
# coding: utf-8

"""
ResNet-50 backend configuration (weighted-loss variant).

Re-exports the shared top-level config and layers on ResNet-specific
paths and hyperparameters. Modules in this subpackage import this as
``from cardiac_acr.backends.resnet import config as cg`` and reach
both shared (``TRAIN_DIR``, ``CLASS_NAMES``, ``IMAGENET_MEAN``, …) and
backend-specific (``MODEL_DIR``, ``CROSS_VAL_*``, stats spreadsheets)
constants through a single name.

Outputs are nested under a ``Weighted_Loss/`` subfolder so they don't
collide with the UNI backend's outputs in the same ``data/`` tree.
"""

import os

from cardiac_acr.config import *  # noqa: F401,F403  (re-export shared constants)
from cardiac_acr.config import BACKEND_DIR, DATA_DIR, WSI_DIR


#####################################################################
# ResNet output locations (nested under Weighted_Loss/)
#####################################################################

_VARIANT = "Weighted_Loss"

MODEL_DIR = os.path.join(DATA_DIR, "Saved_Models", _VARIANT)
SAVED_DATABASE_DIR = os.path.join(BACKEND_DIR, "Saved_Databases", _VARIANT)
ANNOTATED_PNG_DIR = os.path.join(BACKEND_DIR, "Annotated_Test_Slides", _VARIANT)
SLIDE_DX_DIR = os.path.join(BACKEND_DIR, "Slide_Dx", _VARIANT)
TEST_SLIDE_PREDICTIONS_DIR = os.path.join(BACKEND_DIR, "Test_Slide_Predictions", _VARIANT)
TEST_SLIDE_ANNOTATIONS_DIR = os.path.join(WSI_DIR, "TEST_SLIDE_ANNOTATIONS", _VARIANT)


#####################################################################
# 5-fold cross-validation workspace
#####################################################################

CROSS_VAL_DIR = os.path.join(DATA_DIR, "Cross_Validation")
CROSS_VAL_TRAIN_DIR = os.path.join(CROSS_VAL_DIR, "Training_Sets", "Training")
CROSS_VAL_VALID_DIR = os.path.join(CROSS_VAL_DIR, "Training_Sets", "Validation")
CROSS_VAL_MODEL_DIR = os.path.join(CROSS_VAL_DIR, "Saved_Models")


#####################################################################
# Per-threshold slide-level CSV outputs consumed by stats/
#####################################################################

TRAIN_SET_ANALYSIS_DIR = SLIDE_DX_DIR + "_Training_Set_Analysis"
TEST_SET_ANALYSIS_DIR = SLIDE_DX_DIR + "_Test_Set_Analysis"


#####################################################################
# Pathologist ground-truth and summary spreadsheets
#####################################################################

SPREADSHEETS_DIR = os.path.join(DATA_DIR, "Spreadsheets")
TRAIN_DX_CSV = os.path.join(SPREADSHEETS_DIR, "Training_Set_Pathologist_Dx.csv")
TEST_DX_CSV = os.path.join(SPREADSHEETS_DIR, "Test_Set_Pathologist_Dx.csv")
TRAIN_SUMMARY_CSV = os.path.join(SPREADSHEETS_DIR, "Training_Set_Threshold_Analysis.csv")
TEST_SUMMARY_CSV = os.path.join(SPREADSHEETS_DIR, "Test_Set_Threshold_Analysis.csv")


#####################################################################
# Patch-level predictions dumped by stats/dump_training_predictions.py
#####################################################################

TRAIN_SET_PREDICTIONS_PICKLE = os.path.join(
    SAVED_DATABASE_DIR, "Training_Set_Model_Predictions.pickle"
)


#####################################################################
# Training hyperparameters (defaults — individual scripts may override)
#####################################################################

TRAIN_INPUT_SIZE = 224
TRAIN_BATCH_SIZE = 50
TRAIN_LEARNING_RATE = 5e-4
TRAIN_NUM_EPOCHS = 20
TRAIN_DEFAULT_MODEL_NAME = "resnet50"
