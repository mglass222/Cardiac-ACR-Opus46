#!/usr/bin/env python
# coding: utf-8

"""
Shared paths and constants for the Cardiac-ACR patch/WSI pipeline.

Backend-agnostic. Backend-specific hyperparameters (UNI backbone dims,
head learning rate, ResNet input size, …) live under
``cardiac_acr.backends.<name>.config``.

All inputs and outputs live under ``DATA_DIR`` (``<PROJECT_ROOT>/data``).
"""

import os


### Project Root ###
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#####################################################################
# Preprocessing / slide + patch pipeline paths
#####################################################################

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BACKEND_DIR = os.path.join(DATA_DIR, "Backend")
DEEP_HISTO_DIR = os.path.join(DATA_DIR, "DeepHistoPath")
WSI_DIR = os.path.join(DATA_DIR, "WSI")

PNG_SLIDE_DIR = os.path.join(DEEP_HISTO_DIR, "training_png")
TILE_DATA_DIR = os.path.join(DEEP_HISTO_DIR, "tile_data")
TILE_DIR = os.path.join(DEEP_HISTO_DIR, "tiles_png")
FILTERED_IMAGE_DIR = os.path.join(DEEP_HISTO_DIR, "filter_png")
TISSUE_PERCENT_DIR = os.path.join(DEEP_HISTO_DIR, "display_tissue_percent")
SPLIT_TILE_DIR = os.path.join(DEEP_HISTO_DIR, "tiles_png_split")

TEST_SLIDE_DIR = os.path.join(WSI_DIR, "Test")
TRAIN_SLIDE_DIR = os.path.join(WSI_DIR, "Training")

### Font path (only used if tile summary drawing is re-enabled) ###
FONT_PATH = os.path.join(PROJECT_ROOT, "fonts", "arial.ttf")

### OpenSlide (Windows only — ignored on other platforms) ###
OPENSLIDE_BIN_PATH = os.environ.get("OPENSLIDE_BIN_PATH", "")

ANNOTATION_SIZE = 224
PATCH_SIZE = 224
SCALE_FACTOR = 40

# Minimum probability required for a patch prediction to count.
PREDICTION_THRESHOLD = 0.99
BATCH_SIZE = 200

# For dilating 1R2 patches in count_1r2.py
_1R2_DILATION_ITERS = 28


#####################################################################
# Patch library (produced by extract_patches + create_training_sets)
#####################################################################

PATCH_DIR = os.path.join(DATA_DIR, "Patches")
OPENSLIDE_DIR = os.path.join(PATCH_DIR, "Openslide_Output")

TRAINING_PATCH_DIR = os.path.join(PATCH_DIR, "Training_Sets")
TRAIN_DIR = os.path.join(TRAINING_PATCH_DIR, "Training")
VALID_DIR = os.path.join(TRAINING_PATCH_DIR, "Validation")


#####################################################################
# WSI-level diagnosis output locations
#####################################################################
# Per-backend output paths (SAVED_DATABASE_DIR, SLIDE_DX_DIR,
# ANNOTATED_PNG_DIR, TEST_SLIDE_PREDICTIONS_DIR,
# TEST_SLIDE_ANNOTATIONS_DIR) live in each backend's own config so two
# backends can write into the same data/ tree without collision. They
# are surfaced on the BackendClassifier and consumed by wsi/diagnose
# and the annotate/CSV helpers via explicit arguments.

### Count 1R2 directories ###
COUNT_1R2_DIR = os.path.join(BACKEND_DIR, "Count_1R2")
ROI_1R2_DIR = os.path.join(COUNT_1R2_DIR, "ROI-1R2-Only")
ROI_FILTER_DIR = os.path.join(COUNT_1R2_DIR, "ROI-Filtered-PNG")
ANNOTATED_1R2_DIR = os.path.join(COUNT_1R2_DIR, "Annotated_1R2")
SEGMENTED_DIR = os.path.join(COUNT_1R2_DIR, "Segmented")
BOUNDING_BOXES_DIR = os.path.join(SEGMENTED_DIR, "Bounding_Boxes")
COMBINED_BOXES_DIR = os.path.join(SEGMENTED_DIR, "Combined_Boxes")


#####################################################################
# Classes
#####################################################################

# Ordered to match alphabetical folder order used by torchvision.datasets.ImageFolder.
CLASS_NAMES = ["1R1A", "1R2", "Healing", "Hemorrhage", "Normal", "Quilty"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


#####################################################################
# Image preprocessing — ImageNet normalization (used by every backend)
#####################################################################

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
