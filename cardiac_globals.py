#!/usr/bin/env python
# coding: utf-8


### Main Source / Output Folders ###
OPENSLIDE_BIN_PATH = "C:\\Openslide_4003\\bin\\"
BASE_DIR = "D:\\Cardiac_ACR\\"
BACKEND_DIR = BASE_DIR + "Backend\\"
DEEP_HISTO_DIR = "E:\\Cardiac_ACR\\DeepHistoPath\\"
WSI_DIR = BASE_DIR + "\\WSI\\"

PNG_SLIDE_DIR = DEEP_HISTO_DIR + "training_png\\"
TILE_DATA_DIR = DEEP_HISTO_DIR + "tile_data\\"
TILE_DIR = DEEP_HISTO_DIR + "tiles_png\\"
FILTERED_IMAGE_DIR = DEEP_HISTO_DIR + "filter_png\\"
TISSUE_PERCENT_DIR = DEEP_HISTO_DIR + "display_tissue_percent\\"
SPLIT_TILE_DIR = DEEP_HISTO_DIR + "tiles_png_split\\"


TEST_SLIDE_DIR = BASE_DIR + "WSI\\Test\\"
TRAIN_SLIDE_DIR = BASE_DIR + "WSI\\Training\\"

# Stuff used in front end
PATCHES_DIR = BASE_DIR + "Patches\\"
OPENSLIDE_DIR = PATCHES_DIR + "Openslide_Output\\"
TRAINING_SET_DIR = PATCHES_DIR = "Training_Sets\\"

## Font path for displaying tissue percentage on images ###
FONT_PATH = "C:\\Windows\\Fonts\\Arial.ttf"

ANNOTATION_SIZE = 224
PATCH_SIZE = 224
SCALE_FACTOR = 40

# minimum probability required for model predictions to count
PREDICTION_THRESHOLD = 0.99
BATCH_SIZE = 200

# For dilating 1R2 patches in module "count_1r2.py"
_1R2_DILATION_ITERS = 28


   
##################### DIRECTORIES FOR BACKEND - WEIGHTED LOSS FXN ######################

MODEL_DIR = BASE_DIR + "Saved_Models\\Weighted_Loss\\"
SAVED_DATABASE_DIR = BACKEND_DIR + "Saved_Databases\\Weighted_Loss\\"
ANNOTATED_PNG_DIR = BACKEND_DIR + "Annotated_Test_Slides\\Weighted_Loss\\"
SLIDE_DX_DIR = BACKEND_DIR + "Slide_Dx\\Weighted_Loss\\"
TEST_SLIDE_PREDICTIONS_DIR = BACKEND_DIR + "Test_Slide_Predictions\\Weighted_Loss\\"
TEST_SLIDE_ANNOTATIONS_DIR =  WSI_DIR + "TEST_SLIDE_ANNOTATIONS\\Weighted_Loss\\"


####### DIRECTORIES FOR WEIGHTED LOSS FXN - TRAINING SET ANALYSIS #########

# MODEL_DIR = BASE_DIR + "Saved_Models\\Weighted_Loss\\"
# SAVED_DATABASE_DIR = BACKEND_DIR + "Saved_Databases\\Weighted_Loss_Training_Set\\"
# SLIDE_DX_DIR = BACKEND_DIR + "Slide_Dx\\Weighted_Loss_Training_Set_Analysis\\"
# TEST_SLIDE_PREDICTIONS_DIR = BACKEND_DIR + "Test_Slide_Predictions\\Weighted_Loss_Training_Set\\"


####### DIRECTORIES FOR WEIGHTED LOSS FXN - TEST SET ANALYSIS #########

# MODEL_DIR = BASE_DIR + "Saved_Models\\Weighted_Loss\\"
# SAVED_DATABASE_DIR = BACKEND_DIR + "Saved_Databases\\Weighted_Loss\\"
# SLIDE_DX_DIR = BACKEND_DIR + "Slide_Dx\\Weighted_Loss_Test_Set_Analysis\\"
# TEST_SLIDE_PREDICTIONS_DIR = BACKEND_DIR + "Test_Slide_Predictions\\Weighted_Loss\\"


