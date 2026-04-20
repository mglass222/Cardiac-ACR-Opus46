#!/usr/bin/env python
# coding: utf-8


# Import Modules, etc.

import os
from os import listdir
from os.path import isfile, isdir, join

import glob
import pickle
import shutil
import math
import multiprocessing

from PIL import Image

import time


from cardiac_acr import cardiac_globals as cg
from cardiac_acr import cardiac_utils as utils


# make sure directory exists
utils.make_directory(cg.TILE_DIR)

utils.make_directory(cg.SPLIT_TILE_DIR)



def process_tilesets_multiprocess(slide_num):

    t = time.time()

    slide_num = utils.pad_image_number(slide_num)

    print(f"slide_num = {slide_num}")

    print(f"Splitting tiles from slide {slide_num} into patches")

    process_tiles(slide_num)

    print("Done with slide ", slide_num, time.time() - t)



def process_tiles(slide_num):

    TILE_SET_DIR = os.path.join(cg.TILE_DIR, str(slide_num))

    OUTPUT_DIR = os.path.join(cg.SPLIT_TILE_DIR, str(slide_num))

    utils.make_directory(OUTPUT_DIR)

    tile_list = listdir(TILE_SET_DIR)
    
    # how many processes to use
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    
    num_tiles = len(tile_list)
    
    print("Number of processes: " + str(num_processes))
    print("Number of tiles: " + str(num_tiles))

    if num_processes > num_tiles:
        num_processes = num_tiles
        
    images_per_process = num_tiles / num_processes
    
    print("images per process = ", images_per_process)
    
    # each task specifies a range of tiles
    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        if num_tiles is not None:
            sublist = tile_list[start_index - 1:end_index]
            tasks.append((sublist, slide_num))
            print("Task #" + str(num_process) + ": Process " + str(len(sublist)) + " Tiles")
        else:
            print("Error, got an empty tileset")

  # start tasks
    results = []
    for t in tasks:
        if num_tiles is not None:
            results.append(pool.apply_async(tiles_to_patches, t))
        else:
            print("Error, num_tiles == 0")

    for result in results:
        if num_tiles is not None:
            print("Getting results")
            tiles = result.get()
            print(f"Done converting {len(tiles)} tiles:")
            	
    pool.close()
    pool.join()


# Split the tiles into 224x224 patches for classification.

def tiles_to_patches(tile_list, slide_num):
    
    # print("Made it into tiles_to_patches")
    
    t = time.time()
    
    TILE_SET_DIR = os.path.join(cg.TILE_DIR, str(slide_num))
    OUTPUT_DIR = os.path.join(cg.SPLIT_TILE_DIR, str(slide_num))
    
    # print("Tile list = ", tile_list)
    
    for i in range(len(tile_list)):
        tile = tile_list[i]

        tile_path = os.path.join(TILE_SET_DIR, tile)
        tile_name = os.path.splitext(os.path.basename(tile_path))[0] + ".png"

        image = Image.open(tile_path)

        image_width, image_height = image.size
        
        # PATCH_SIZE = cg.PATCH_SIZE

        PATCH_SIZE = cg.ANNOTATION_SIZE

        if image_width % PATCH_SIZE == 0:
            x_steps = int(image_width/PATCH_SIZE)
        
        else:
            x_steps = int(image_width/PATCH_SIZE) + 1

        if image_height % PATCH_SIZE == 0:
            y_steps = int(image_height/PATCH_SIZE)
        
        else:
            y_steps = int(image_height/PATCH_SIZE) + 1

        # print("x_steps = ",x_steps)
        # print("y_steps = ", y_steps)
        # print("total steps = ", x_steps*y_steps)
        
        y_break = False
        for y in range(y_steps):
            if not y_break:
                y_start = y * PATCH_SIZE
                y_stop = min((y_start + PATCH_SIZE), image_height)
                if y_stop == image_height :
                    y_break = True

                x_break = False    
                for x in range(x_steps):
                    if not x_break:
                        x_start = x * PATCH_SIZE
                        x_stop = min((x_start + PATCH_SIZE), image_width)
                        if x_stop == image_width:
                            x_break = True

                        new_image = image.crop((x_start, y_start, x_stop, y_stop))

                        if new_image.size[0] < PATCH_SIZE or new_image.size[1] < PATCH_SIZE:
                            # print("Padding image")
                            padded_image = Image.new('RGB', (PATCH_SIZE,PATCH_SIZE),'Black')
                            padded_image.paste(new_image)
                            new_image = padded_image

                        new_image_name = utils.get_patchname(tile_name, slide_num, x_start, y_start)

                        new_image_path = os.path.join(OUTPUT_DIR, str(new_image_name))
                        # print(new_image_path)
                        new_image.save(new_image_path)


        
    return tile_list
