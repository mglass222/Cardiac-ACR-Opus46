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


##################### TRAINING / CROSS-VALIDATION / STATS ##############################

### Patch library (class-indexed PNGs produced by cardiac_acr/training/extract_patches.py) ###
PATCH_DIR = os.path.join(DATA_DIR, "Patches")
OPENSLIDE_DIR = os.path.join(PATCH_DIR, "Openslide_Output")

### Train/Val split used by cardiac_acr/training/train.py ###
TRAINING_PATCH_DIR = os.path.join(PATCH_DIR, "Training_Sets")
TRAIN_DIR = os.path.join(TRAINING_PATCH_DIR, "Training")
VALID_DIR = os.path.join(TRAINING_PATCH_DIR, "Validation")

### 5-fold cross validation workspace (cardiac_acr/training/cross_validation.py) ###
CROSS_VAL_DIR = os.path.join(DATA_DIR, "Cross_Validation")
CROSS_VAL_TRAIN_DIR = os.path.join(CROSS_VAL_DIR, "Training_Sets", "Training")
CROSS_VAL_VALID_DIR = os.path.join(CROSS_VAL_DIR, "Training_Sets", "Validation")
CROSS_VAL_MODEL_DIR = os.path.join(CROSS_VAL_DIR, "Saved_Models")

### Per-threshold slide-level CSV outputs consumed by cardiac_acr/stats/ ###
TRAIN_SET_ANALYSIS_DIR = os.path.join(SLIDE_DX_DIR + "_Training_Set_Analysis")
TEST_SET_ANALYSIS_DIR = os.path.join(SLIDE_DX_DIR + "_Test_Set_Analysis")

### Pathologist ground-truth and summary spreadsheets used by stats scripts ###
SPREADSHEETS_DIR = os.path.join(DATA_DIR, "Spreadsheets")
TRAIN_DX_CSV = os.path.join(SPREADSHEETS_DIR, "Training_Set_Pathologist_Dx.csv")
TEST_DX_CSV = os.path.join(SPREADSHEETS_DIR, "Test_Set_Pathologist_Dx.csv")
TRAIN_SUMMARY_CSV = os.path.join(SPREADSHEETS_DIR, "Training_Set_Threshold_Analysis.csv")
TEST_SUMMARY_CSV = os.path.join(SPREADSHEETS_DIR, "Test_Set_Threshold_Analysis.csv")

### Patch-level predictions dumped by cardiac_acr/stats/dump_training_predictions.py ###
TRAIN_SET_PREDICTIONS_PICKLE = os.path.join(
    SAVED_DATABASE_DIR, "Training_Set_Model_Predictions.pickle"
)

### Class definitions (patch-level, 6 classes) ###
# Ordered to match alphabetical folder order used by torchvision.datasets.ImageFolder.
CLASS_NAMES = ["1R1A", "1R2", "Healing", "Hemorrhage", "Normal", "Quilty"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

### Training hyperparameters (defaults — individual scripts may override) ###
TRAIN_INPUT_SIZE = 224
TRAIN_BATCH_SIZE = 50
TRAIN_LEARNING_RATE = 5e-4
TRAIN_NUM_EPOCHS = 20
TRAIN_DEFAULT_MODEL_NAME = "resnet50"

### ImageNet normalization used during training + inference ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
