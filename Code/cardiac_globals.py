#!/usr/bin/env python
# coding: utf-8

import os

### Project Root ###
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

### Main Source / Output Folders ###
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

## Font path ###
FONT_PATH = os.path.join(PROJECT_ROOT, "fonts", "arial.ttf")

## OpenSlide (Windows only — ignored on other platforms) ###
OPENSLIDE_BIN_PATH = os.environ.get("OPENSLIDE_BIN_PATH", "")

ANNOTATION_SIZE = 224
PATCH_SIZE = 224
SCALE_FACTOR = 40

# minimum probability required for model predictions to count
PREDICTION_THRESHOLD = 0.99
BATCH_SIZE = 200

# For dilating 1R2 patches in module "count_1r2.py"
_1R2_DILATION_ITERS = 28


##################### DIRECTORIES FOR BACKEND - WEIGHTED LOSS FXN ######################

MODEL_DIR = os.path.join(DATA_DIR, "Saved_Models", "Weighted_Loss")
SAVED_DATABASE_DIR = os.path.join(BACKEND_DIR, "Saved_Databases", "Weighted_Loss")
ANNOTATED_PNG_DIR = os.path.join(BACKEND_DIR, "Annotated_Test_Slides", "Weighted_Loss")
SLIDE_DX_DIR = os.path.join(BACKEND_DIR, "Slide_Dx", "Weighted_Loss")
TEST_SLIDE_PREDICTIONS_DIR = os.path.join(BACKEND_DIR, "Test_Slide_Predictions", "Weighted_Loss")
TEST_SLIDE_ANNOTATIONS_DIR = os.path.join(WSI_DIR, "TEST_SLIDE_ANNOTATIONS", "Weighted_Loss")

### Count 1R2 directories ###
COUNT_1R2_DIR = os.path.join(BACKEND_DIR, "Count_1R2")
ROI_1R2_DIR = os.path.join(COUNT_1R2_DIR, "ROI-1R2-Only")
ROI_FILTER_DIR = os.path.join(COUNT_1R2_DIR, "ROI-Filtered-PNG")
ANNOTATED_1R2_DIR = os.path.join(COUNT_1R2_DIR, "Annotated_1R2")
SEGMENTED_DIR = os.path.join(COUNT_1R2_DIR, "Segmented")
BOUNDING_BOXES_DIR = os.path.join(SEGMENTED_DIR, "Bounding_Boxes")
COMBINED_BOXES_DIR = os.path.join(SEGMENTED_DIR, "Combined_Boxes")
