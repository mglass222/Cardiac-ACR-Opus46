#!/usr/bin/env python
# coding: utf-8

"""
Extract annotated training patches from whole-slide images.

For each ``.svs`` slide in ``cg.TRAIN_SLIDE_DIR`` with a matching
ImageScope-style ``.xml`` annotation file, this script parses the XML,
groups ``Region`` elements by their ``Name`` attribute (the class label),
and uses OpenSlide to crop the bounding rectangle of each region at
level 0. The cropped patch is saved as a PNG under:

    cg.OPENSLIDE_DIR / <class> / slide_<slide_num>_<class>_region_id_<id>.png

This produces the class-indexed patch library that
``create_training_sets`` then splits into Training / Validation, and
that ``encode_patches`` runs through UNI2-h.

Ported from the ``Extract_Patches_V5.ipynb`` notebook.

Usage:
    python -m cardiac_acr.preprocessing.extract_patches
"""

import os
import pickle
import time
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isdir

import numpy as np

from cardiac_acr import config as cg
from cardiac_acr.preprocessing.openslide_compat import openslide


# Default class labels to pull out of each XML file. Each value must match
# the ``Name`` attribute on the ``Annotation`` element in ImageScope.
DEFAULT_EXTRACT_TYPES = list(cg.CLASS_NAMES)


def make_directory(directory):
    """Create ``directory`` (and parents) if it does not already exist."""
    if not os.path.exists(directory):
        print(f"Output directory doesn't exist, will create:\n  {directory}")
        os.makedirs(directory)


def get_xml_files(slide_dir):
    """
    Walk ``slide_dir`` and return matching slide/XML lists.

    Returns
    -------
    slides : list[str]
        All ``.svs`` filenames found in ``slide_dir``.
    xml_files : list[str]
        Sorted list of ``.xml`` files whose corresponding ``.svs`` slide
        exists in ``slides``. The ``template.xml`` placeholder is skipped.
    """
    print(slide_dir)

    files = listdir(slide_dir)

    slides = [f for f in files if f.lower().endswith(".svs")]

    xml_files = []
    for f in files:
        if not f.lower().endswith(".xml"):
            continue
        stem = f.rsplit(".", 1)[0]
        if stem == "template":
            continue
        if stem + ".svs" in slides:
            xml_files.append(f)

    return slides, sorted(xml_files)


def parse_xml_file(slide_dir, xml_file, extract_types):
    """
    Parse a single ImageScope XML file and group Regions by class name.

    Parameters
    ----------
    slide_dir : str
        Directory containing ``xml_file``.
    xml_file : str
        Filename of the XML annotation (not a full path).
    extract_types : iterable[str]
        Class labels to collect. Each corresponds to the ``Name``
        attribute on an ``Annotation`` element.

    Returns
    -------
    all_regions : list[list[Element]]
        ``all_regions[i]`` is the list of Region elements for
        ``extract_types[i]``.
    all_vertices : list[list[Element]]
        Matching Vertex elements (4 per region, in XML order).
    """
    if xml_file == "template.xml":
        return [], []

    tree = ET.parse(os.path.join(slide_dir, xml_file))
    root = tree.getroot()

    all_regions, all_vertices = [], []

    for item in extract_types:
        regions = root.findall(f".//*[@Name='{item}']//Region")
        vertices = root.findall(f".//*[@Name='{item}']//Vertex")
        all_regions.append(regions)
        all_vertices.append(vertices)

    return all_regions, all_vertices


def read_patch(slide_ptr, region_vertices):
    """
    Crop the axis-aligned bounding box of a 4-vertex Region from the slide.

    Parameters
    ----------
    slide_ptr : openslide.OpenSlide
        Open slide handle.
    region_vertices : list[Element]
        The 4 Vertex elements that define the region rectangle.

    Returns
    -------
    PIL.Image.Image
        The cropped patch converted to RGB.
    """
    vertex_coords = np.zeros([len(region_vertices), 2])
    for k, vertex in enumerate(region_vertices):
        vertex_coords[k, 0] = vertex.attrib["X"]
        vertex_coords[k, 1] = vertex.attrib["Y"]

    top_left = (
        int(vertex_coords[:, 0].min()),
        int(vertex_coords[:, 1].min()),
    )
    width = int(vertex_coords[:, 0].max() - vertex_coords[:, 0].min())
    height = int(vertex_coords[:, 1].max() - vertex_coords[:, 1].min())

    image = slide_ptr.read_region(top_left, 0, (width, height))
    return image.convert(mode="RGB")


def get_num_patches(openslide_dir):
    """
    Count the PNGs inside each class subdirectory of ``openslide_dir``.

    Returns a ``dict`` keyed by class name with the patch count as the
    value, and prints the total.
    """
    classes = listdir(openslide_dir)
    patch_dict = {cls: 0 for cls in classes}
    total = 0

    for cls in patch_dict:
        path = os.path.join(openslide_dir, cls)
        if isdir(path):
            count = len(listdir(path))
            patch_dict[cls] = count
            total += count

    print(f"Total number of patches = {total}")
    return patch_dict


def extract_patches(slide_dir=None, openslide_dir=None, extract_types=None):
    """
    Run the full patch-extraction pipeline.

    Parameters
    ----------
    slide_dir : str, optional
        Directory holding the ``.svs`` / ``.xml`` pairs. Defaults to
        ``cg.TRAIN_SLIDE_DIR``.
    openslide_dir : str, optional
        Root directory under which class subfolders of PNG patches are
        written. Defaults to ``cg.OPENSLIDE_DIR``.
    extract_types : list[str], optional
        Class labels to extract. Defaults to ``cg.CLASS_NAMES``.
    """
    if slide_dir is None:
        slide_dir = cg.TRAIN_SLIDE_DIR
    if openslide_dir is None:
        openslide_dir = cg.OPENSLIDE_DIR
    if extract_types is None:
        extract_types = DEFAULT_EXTRACT_TYPES

    slides, xml_files = get_xml_files(slide_dir)
    print(xml_files)

    make_directory(cg.PATCH_DIR)
    make_directory(openslide_dir)

    t0 = time.time()

    for xml_file in xml_files:
        slide_stem = xml_file.rsplit(".", 1)[0]
        slide_name = slide_stem + ".svs"
        slide_path = os.path.join(slide_dir, slide_name)

        slide_ptr = openslide.OpenSlide(str(slide_path))
        print(f"Getting patches from slide: {slide_name}")

        all_regions, all_vertices = parse_xml_file(slide_dir, xml_file, extract_types)

        for patch_type, patch_regions, patch_vertices in zip(
            extract_types, all_regions, all_vertices
        ):
            out_dir = os.path.join(openslide_dir, patch_type)
            make_directory(out_dir)

            for k, region in enumerate(patch_regions):
                # Each region has exactly 4 vertices in ImageScope.
                vertices = patch_vertices[4 * k : 4 * (k + 1)]
                region_id = region.attrib["Id"]

                image_name = (
                    f"slide_{slide_stem.zfill(3)}_{patch_type}"
                    f"_region_id_{region_id}.png"
                )

                image = read_patch(slide_ptr, vertices)
                image.save(os.path.join(out_dir, image_name), "PNG")
                image.close()

        slide_ptr.close()

    elapsed = time.time() - t0
    print(f"\nDone with patch acquisition in {elapsed:.2f} seconds")

    patch_dict = get_num_patches(openslide_dir)
    for key, value in patch_dict.items():
        print(key, value)

    # Persist patch counts for downstream tooling.
    pickle_path = os.path.join(cg.PATCH_DIR, "patch_dict.pickle")
    with open(pickle_path, "wb") as handle:
        pickle.dump(patch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved patch counts to {pickle_path}")


def main():
    """Extract patches for every class in ``cg.CLASS_NAMES``."""
    extract_patches()


if __name__ == "__main__":
    main()
