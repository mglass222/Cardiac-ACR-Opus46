#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
from os import listdir
from os.path import isfile, isdir, join
import xml.etree.ElementTree as ET

# import openslide
import import_openslide

import pickle
import shutil
import multiprocessing

import numpy as np
from PIL import Image
import time

from datetime import datetime
import re

import cardiac_globals as cg



def get_filtered_image_path(slide_number):

    filtered_image = None
    files = listdir(cg.FILTERED_IMAGE_DIR)

    for file in files:
        slide_num = file.split("-")[0]
        status = file.split(".")[0].split("-")[-1]

        if int(slide_num) == slide_number and status == 'filtered':
            filtered_image = file

        else:
            pass

    return filtered_image


# In[ ]:


def make_directory(directory):     
    if os.path.exists(directory):
        pass
    else:
        # print("Output directory doesn't exist, will create:")
        # print(directory)
        os.makedirs(directory)


# In[2]:


def initialize_gpu():
    
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
    
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        print(f"GPU {i} = ", torch.cuda.get_device_name(i))
    
    return device


# In[3]:


# Get SVS from WSI directory
def get_files():
    
    t = time.time()
    slides = []
    files = [file for file in listdir(cg.TEST_SLIDE_DIR) if isfile(join(cg.TEST_SLIDE_DIR, file))]
    
    for i in range(len(files)):
        if (files[i].split(".")[1] == "svs"):
            slides.append(files[i])
    print(f"Done getting pointers to slides {time.time() - t}")

    return slides


# In[4]:


def get_test_slide_numbers():
    
    slides = [file for file in listdir(cg.TEST_SLIDE_DIR) if file.split(".")[1] == "svs"]
    slide_list = []
    # get rid of the .svs and make integer
    for i in range(len(slides)):
        item = slides[i]
        item = int(item.split(".")[0])
        slide_list.append(item)
        
    return slide_list



def get_training_slide_numbers():
    
    slides = [file for file in listdir(cg.TRAIN_SLIDE_DIR) if file.split(".")[1] == "svs"]
    slide_list = []
    # get rid of the .svs and make integer
    for i in range(len(slides)):
        item = slides[i]
        item = str(item.split(".")[0]).zfill(3)
        slide_list.append(item)
        
    return slide_list


# In[5]:


def get_slide_info(): 
    
# make a dictionary of slide names with the slide dimensions for rebuilding the slides later
    t = time.time()
    slides = GetFiles()
    slide_dict = {}
    for i in range(len(slides)):
        slide_name = slides[i]
        slide_ptr = openslide.OpenSlide(cg.TEST_SLIDE_DIR + slide_name)
        slide_width, slide_height = slide_ptr.dimensions
        slide_dict.update({slide_name: (slide_width,slide_height)})
    
    with open(cg.SAVED_DATABASE_DIR + 'slide_dict.pickle', 'wb') as handle:
        pickle.dump(slide_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Done getting slide dimensions {time.time() - t}")
    
    return slide_dict


# In[6]:


def get_patches_dir_from_slide_number(slide_number):

    directory = cg.SPLIT_TILE_DIR + str(slide_number)

    return directory


# In[7]:


def model_prediction_dict_to_csv(slide_number):
    
    """ e.g. "model_predictions_dict_001_filtered.pickle" """
    
    import csv
    import os

    filename = "model_predictions_dict_" + str(slide_number) + "_filtered.pickle"

    
    with open(cg.SAVED_DATABASE_DIR + filename, 'rb') as handle:
        mydict = pickle.load(handle)
    
    # get rid of the ".pickle"
    filename = filename.replace(filename.split(".")[1], "") 
    filename = filename.replace(".", "")
    
    outfile = cg.TEST_SLIDE_PREDICTIONS_DIR + filename + ".csv"

    make_directory(cg.TEST_SLIDE_PREDICTIONS_DIR)

    # delete old file is exists
    if isfile(outfile): 
        print("Removing old prediction dict")
        os.remove(outfile)
    else:
        print("No old prediction dict found")

    # print("CSV filename = ", outfile)
    with open(outfile, 'w', newline='') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])



def slide_dx_to_csv(slide_dx_dict, filename):

    import csv

    try:
        # get rid of the ".pickle"
        filename = filename.split(".")[0] + ".csv"

        outfile = cg.SLIDE_DX_DIR + filename

        # delete old file is exists
        if isfile(outfile): os.remove(outfile)

        # print("CSV filename = ", outfile)
        with open(outfile, 'w', newline='') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in slide_dx_dict.items():
                writer.writerow([key, value])

        print(f"Successfully wrote {outfile}")

    except:

        print(f"Error writing {outfile}")


# In[8]:


def get_png_slide_path(slide_number):
    
    files = listdir(cg.PNG_SLIDE_DIR)
    
    png_slide_path = None
    
    for file in files:
        if str(file.split("-")[0]) == str(slide_number):
            png_slide = file
            png_slide_path = cg.PNG_SLIDE_DIR + png_slide
    
    
    return png_slide_path


# In[9]:


def get_png_slide_name(slide_number):
    
    files = listdir(cg.PNG_SLIDE_DIR)
    
    for file in files:
        if str(file.split("-")[0]) == str(slide_number):
            png_slide_name = file
    
    return png_slide_name


# In[10]:


def get_coords_from_name(name):
    
    # Get coordinates from filename
    m = re.match(".*-x([\d]*)-y([\d]*).*\..*", name)
    x_coord = int(m.group(1))
    y_coord = int(m.group(2))
    
    return x_coord, y_coord


# In[11]:


def parse_dimensions_from_image_filename(filename):
    
    # Get coordinates from filename
    m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
    large_w = int(m.group(1))
    large_h = int(m.group(2))
    small_w = int(m.group(3))
    small_h = int(m.group(4))
    
    return large_w, large_h, small_w, small_h


# In[12]:


def large_to_small_coords(large_w, large_h, small_w, small_h, large_x, large_y):
    
    """ 
    Converts from coordinates of original large image to scaled down image.
    Dont need to include SCALE_FACTOR in calculations since it cancels out in the equations
    and all you need are the ratios of the length and width to calculate
    """
    

    small_x = round(large_x * (small_w/large_w))
    small_y = round(large_y * (small_h/large_h))

    # small_x = int(large_x * (small_w/large_w))
    # small_y = int(large_y * (small_h/large_h))

    
    return small_x, small_y


# In[13]:


def small_to_large_coords(large_w, large_h, small_w, small_h, small_x, small_y):
    
    """ 
    Converts from coordinates of the scaled down image to the original large image.
    Dont need to include SCALE_FACTOR in calculations since it cancels out in the equations
    and all you need are the ratios of the length and width to calculate
    """
    
    large_x = round(small_x * (large_w/small_w))
    large_y = round(small_y * (large_h/small_h))

    # large_x = int(small_x * (large_w/small_w))
    # large_y = int(small_y * (large_h/small_h))


    
    return large_x, large_y


# In[14]:


def pad_image_number(number):
    
    if len(str(number)) < 3:
        number = str(number).zfill(3)  

    return number


# In[15]:


def get_patchname(tile_name, slide_num, x_start, y_start):

    """ 
    Get the X and Y coordinates of a tile from the tilename and calculate patch coordinates 
    """
    
    # Get large_x and large_y coords from the tile we are processing
    m = re.match(".*-r([\d]*)-c([\d]*)-x([\d]*)-y([\d]*).*\..*", tile_name)
    
    large_x = int(m.group(3)) + x_start
    large_y = int(m.group(4)) + y_start
    
    patch_name = tile_name.replace(tile_name.split("-")[4], "x" + str(large_x))
    patch_name = patch_name.replace(tile_name.split("-")[5], "y" + str(large_y))
    
    return patch_name


# In[16]:


def clean_csv_files():

    import operator
    import csv
       
    out_dir = "D:\\Cardiac_ACR\\DeepHistoPath\\tile_data_cleaned\\"    
    csv_files = []
    
    num_lines = 14
    
    items = listdir(cg.TILE_DATA_DIR)
    
    for item in items:
        csv_path = join(cg.TILE_DATA_DIR, item)
        if isfile(csv_path):
            file_name = item.split(".")[0]
            file_ext = item.split(".")[1]

            outname = out_dir + file_name + "_cleaned_sorted" + "." + file_ext

            with open(csv_path, "r") as infile, open(outname, "w") as outfile:
                reader = csv.reader(infile)
                for i in range(num_lines):
                    idk = next(reader)  # skip the headers
                    
                sortedlist = sorted(reader, key=operator.itemgetter(20), reverse=True) 
                
                writer = csv.writer(outfile)
                for row in sortedlist:
                   # process each row
                   writer.writerow(row)
                    
                print("Done writing csv file")


# In[17]:


def make_top_slides_csv_file():
    
    import csv
    skiplines = 14
    
    path = cg.TILE_DATA_DIR

    csv_files = [file for file in listdir(cg.TILE_DATA_DIR) if "filtered" not in file]
#     print(csv_files)
    
    for file in csv_files:
        filename = file.split(".")[0]
        fileext = file.split(".")[1]
        if fileext == "csv":
            outname =  filename + "_filtered.csv"
            
            with open(join(path, file), "r") as infile, open(join(path, outname), "w") as outfile:
                
                writer = csv.writer(outfile)

                for i in range(skiplines):
                    next(infile)

                # remove save tiles with minimum tiles score
                for row in csv.reader(infile):
                    if row[20] >= "0.1" or row[20] == "Score":
                        writer.writerow(row)
                    
                    

def filter_tiles_multiprocess(slide_number):
    
    """ 
    just want to use this to display tissue percent on tiles like we do with patches
    so need to make a tissue percent dict for tiles 
    """
    
    tissue_percent_dict_tiles = filter_tiles.multiprocess_apply_filters_to_images(slide_number, save=False)
    
    print(f"Filtering {len(tissue_percent_dict_tiles)} Tiles")
    
    save_name = "tissue_percent_dict_tiles_" + str(slide_number) + ".pickle"
    
    with open(cg.SAVED_DATABASE_DIR + save_name, 'wb') as handle:
        pickle.dump(tissue_percent_dict_tiles, handle, protocol=pickle.HIGHEST_PROTOCOL)



def display_tissue_percent_patches(slide_number):
    
    """ 
    Display tissue percent on 224x224 patches
    """
    from PIL import Image, ImageFont, ImageDraw, ImageEnhance
    
    print("Creating tissue percent Patches")
    
    filename = "tissue_percent_dict_" + str(slide_number) + ".pickle"
    
    with open(cg.SAVED_DATABASE_DIR + filename, 'rb') as handle:
        tissue_percent_dict_patches = pickle.load(handle)
        
    # patches
    patches_output_dir = TISSUE_PERCENT_DIR + str(slide_number) + "\\patches\\"
    utils.make_directory(patches_output_dir)
    
    print(f"writing patches to {patches_output_dir}")

    # write tissue percent on patches
    i = 0
    for key, value in tissue_percent_dict_patches.items():
        imagepath = key
        percent = round(value, 1)
        image_name = imagepath.split(".")[0].split("\\")[5]
        image_ext = imagepath.split(".")[1]
        
        image = Image.open(imagepath).convert("RGBA")
        
        draw = ImageDraw.Draw(image)
        
        draw.rectangle(((0, 0), (100, 50)), fill="green")
        draw.text((5, 5), str(percent), font = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", 30))

        filename = str(image_name) + "." + str(image_ext)
        image_path = os.path.join(patches_output_dir, filename)
        
        image = image.convert("RGB")
        image.save(image_path, "JPEG")
        
        if (i % 500 == 0):
            print(f"Processed {i} out of {len(tissue_percent_dict_patches)} Patches")
            
        i += 1
    

def display_tissue_percent_tiles(slide_number):

	""" 
	Display tissue percent on large tiles
	"""

	from PIL import Image, ImageFont, ImageDraw, ImageEnhance

	print("Creating tissue percent Tiles")
	    
	filename = "tissue_percent_dict_tiles_" + str(slide_number) + ".pickle"

	with open(cg.SAVED_DATABASE_DIR + filename, 'rb') as handle:
	    tissue_percent_dict_tiles = pickle.load(handle)


	 # tiles
	tiles_output_dir = TISSUE_PERCENT_DIR + str(slide_number) + "\\tiles\\"
	utils.make_directory(tiles_output_dir)

	print(f"writing tiles to {tiles_output_dir}")
	    
	# write tissue percent on tiles
	i = 0
	for key, value in tissue_percent_dict_tiles.items():
	    imagepath = key
	    percent = round(value, 1)
	    image_name = imagepath.split(".")[0].split("\\")[5]
	    image_ext = imagepath.split(".")[1]
	    
	    image = Image.open(imagepath).convert("RGBA")
	    
	    draw = ImageDraw.Draw(image)
	    
	    draw.rectangle(((0, 0), (400, 200)), fill="green")
	    draw.text((5, 5), str(percent), font = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", 200))

	    filename = str(image_name) + "." + str(image_ext)
	    image_path = os.path.join(tiles_output_dir, filename)
	    
	    image = image.convert("RGB")
	    image.save(image_path, "JPEG")
	    
	    if (i % 100 == 0):
	        print(f"Processed {i} out of {len(tissue_percent_dict_tiles)} Tiles")
	        
	    i += 1    


