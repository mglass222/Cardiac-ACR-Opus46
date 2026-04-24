#!/usr/bin/env python
# coding: utf-8

"""
Split the extracted patch library into training and validation folders.

``extract_patches.py`` writes PNG patches into
``cg.OPENSLIDE_DIR/<class>/``. This script copies those patches into the
``Training_Sets`` layout that ``train.py`` (and
``torchvision.datasets.ImageFolder``) expects:

    cg.TRAIN_DIR / <class> / *.png
    cg.VALID_DIR / <class> / *.png

Split policy: a patch belongs to the training set iff its source slide
number (parsed from the ``slide_<num>_...`` filename produced by
``extract_patches.read_patch``) is in ``TRAIN_SLIDES``. Everything else
goes to validation. The default ``TRAIN_SLIDES`` list is the exact
80/20 assignment used by the original paper and is preserved verbatim
from ``Create_Training_Sets_V8.ipynb``.

Usage:
    python -m cardiac_acr.preprocessing.create_training_sets
"""

import os
import random
import shutil
from os import listdir
from os.path import isdir, join

from cardiac_acr import config as cg
from cardiac_acr.preprocessing import preprocess_data_utils as data_utils


# Reproduces the exact train/val assignment from
# Create_Training_Sets_V8.ipynb so the published split can be recreated.
TRAIN_SLIDES = [
    "253", "299", "214", "053", "062", "069", "310", "217", "319", "079",
    "295", "318", "080", "031", "056", "035", "098", "150", "235", "006",
    "037", "092", "239", "066", "042", "097", "215", "013", "249", "032",
    "254", "237", "059", "247", "102", "311", "220", "034", "243", "033",
    "016", "106", "196", "250", "095", "210", "030", "014", "246", "041",
    "010", "090", "004", "085", "071", "304", "229", "192", "024", "003",
    "216", "206", "078", "312", "110", "011", "005", "019", "303", "044",
    "290", "057", "317", "073", "100", "281", "286", "055", "036", "208",
    "201", "072", "204", "306", "015", "045", "064", "300", "039", "198",
    "065", "191", "101", "107", "087", "294", "296", "197", "091", "084",
    "205", "048", "228", "279", "082", "089", "221", "309", "251", "256",
    "241", "093", "043", "077", "051", "105", "209", "020", "285", "001",
    "099", "244", "302", "088", "054", "301", "293", "190", "211", "307",
    "316", "284", "288", "009", "063", "194", "213", "083", "212", "193",
    "008", "282", "226", "068", "200", "203", "007", "236", "104", "075",
    "264", "305", "278", "067", "012", "074", "195", "096", "297", "283",
    "238", "046", "108", "289",
]

IDEAL_SPLIT = 0.8


def _reset_directory(directory):
    """Ensure ``directory`` exists and is empty (destructive)."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def _slide_num_from_filename(filename):
    """Pull the zero-padded slide number out of a patch filename.

    Patch filenames produced by ``extract_patches.py`` have the form
    ``slide_<num>_<class>_region_id_<id>.png``.
    """
    return filename.split(".")[0].split("_")[1]


def slide_assignments(slide_dir=None, fraction=IDEAL_SPLIT, seed=None):
    """Randomly assign 80% of slides to the training split.

    Returns
    -------
    list[str]
        Slide stems (without extension) chosen for training.
    """
    if slide_dir is None:
        slide_dir = cg.TRAIN_SLIDE_DIR

    slide_names = [
        f.rsplit(".", 1)[0]
        for f in listdir(slide_dir)
        if f.lower().endswith(".svs")
    ]

    num_train = round(len(slide_names) * fraction)

    rng = random.Random(seed) if seed is not None else random
    return rng.sample(slide_names, num_train)


def hypothetical_percentages(slide_num, openslide_dir=None):
    """Project train-set share per class if slides ``<= slide_num`` go to train.

    Useful when picking a numeric cutoff for the split. Operates purely
    on filenames inside ``openslide_dir``.
    """
    if openslide_dir is None:
        openslide_dir = cg.OPENSLIDE_DIR

    percent_dict = {}
    for item in listdir(openslide_dir):
        path = join(openslide_dir, item)
        if not isdir(path):
            continue
        patches = listdir(path)
        train_files = [f for f in patches if int(_slide_num_from_filename(f)) <= slide_num]
        total = len(patches)
        if total == 0:
            continue
        percent_dict[item] = round(len(train_files) / total, 2)

    return percent_dict


def create_training_sets(train_slides=None, classes=None,
                         openslide_dir=None, train_dir=None, valid_dir=None):
    """Copy patches from ``openslide_dir`` into ``train_dir`` / ``valid_dir``.

    Destructive — both split directories are wiped before being
    repopulated so repeated runs are idempotent.
    """
    if train_slides is None:
        train_slides = TRAIN_SLIDES
    if openslide_dir is None:
        openslide_dir = cg.OPENSLIDE_DIR
    if train_dir is None:
        train_dir = cg.TRAIN_DIR
    if valid_dir is None:
        valid_dir = cg.VALID_DIR

    if classes is None or classes == "all":
        classes = listdir(openslide_dir)

    train_slides = set(train_slides)

    _reset_directory(cg.TRAINING_PATCH_DIR)
    _reset_directory(train_dir)
    _reset_directory(valid_dir)

    for item in classes:
        src_dir = join(openslide_dir, item)
        if not isdir(src_dir):
            continue

        all_files = listdir(src_dir)
        train_files = [f for f in all_files if _slide_num_from_filename(f) in train_slides]
        val_files = [f for f in all_files if f not in set(train_files)]

        train_dst = join(train_dir, item)
        val_dst = join(valid_dir, item)
        _reset_directory(train_dst)
        _reset_directory(val_dst)

        print("Getting Training Files for:", item)
        for file in train_files:
            shutil.copy(join(src_dir, file), train_dst)

        print("Getting Validation Files for:", item)
        for file in val_files:
            shutil.copy(join(src_dir, file), val_dst)

    print("\nDONE\n")


def main():
    create_training_sets(TRAIN_SLIDES, classes="all")

    print("Verifying train percentages...\n")
    percent_dict = data_utils.get_percentages()
    print(percent_dict)


if __name__ == "__main__":
    main()
