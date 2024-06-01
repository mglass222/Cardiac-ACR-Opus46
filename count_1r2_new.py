#!/usr/bin/env python
# coding: utf-8

# In[77]:


import sys  
deep_histo_path = "D:\\Python\\"
sys.path.insert(0, deep_histo_path)

import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import random
import os
from os.path import isdir, isfile

import pickle


from PIL import Image, ImageOps
from skimage import filters, io
from scipy import ndimage
from skimage import measure


import cardiac_utils as utils
import cardiac_globals as cg

filtered_image_dir = cg.FILTERED_IMAGE_DIR

count_1r2_dir = 'D:\\Cardiac_ACR\\Backend\\Count_1R2\\'
roi_1r2_dir = 'D:\\Cardiac_ACR\\Backend\\Count_1R2\\ROI-1R2-Only\\'
roi_filter_dir = 'D:\\Cardiac_ACR\\Backend\\Count_1R2\\ROI-Filtered-PNG\\'
annotated_1r2_dir = 'D:\\Cardiac_ACR\\Backend\\Count_1R2\\Annotated_1R2\\'

segmented_dir = 'D:\\Cardiac_ACR\\Backend\\Count_1R2\\Segmented\\'
bounding_boxes_dir = 'D:\\Cardiac_ACR\\Backend\\Count_1R2\\Segmented\\Bounding_Boxes\\'
combined_boxes_dir = 'D:\\Cardiac_ACR\\Backend\\Count_1R2\\Segmented\\Combined_Boxes\\'


# make sure roi save directories exist
if not os.path.isdir(roi_1r2_dir): os.makedirs(roi_1r2_dir)

if not os.path.isdir(roi_filter_dir): os.makedirs(roi_filter_dir)

if not os.path.isdir(combined_boxes_dir):  os.makedirs(combined_boxes_dir)

if not os.path.isdir(segmented_dir):  os.makedirs(segmented_dir)




def combine_boxes(box1, box2):
    
    b1x1 = int(box1[0])
    b1y1 = int(box1[1])
    b1x2 = int(box1[2])
    b1y2 = int(box1[3])
    
    b2x1 = int(box2[0])
    b2y1 = int(box2[1])
    b2x2 = int(box2[2])
    b2y2 = int(box2[3])
    
    minx = min(b1x1,b1x2,b2x1,b2x2)
    maxx = max(b1x1,b1x2,b2x1,b2x2)
    miny = min(b1y1,b1y2,b2y1,b2y2)
    maxy = max(b1y1,b1y2,b2y1,b2y2)
    
    new_box = [minx, miny, maxx, maxy]
    
    return new_box



def analyze_boxes(bounding_boxes):
    
    combined_boxes = []
    combined_boxes_final = []

    for i in range(len(bounding_boxes)):
        box1 = bounding_boxes[i]
        for j in range(len(bounding_boxes)):
            box2 = bounding_boxes[j]
            overlap = check_overlap(box1, box2)
            if overlap:
                box1 = combine_boxes(box1,box2)
        
        combined_boxes.append(box1)
        
    # repeat once more to consolidate boxes
    for i in range(len(combined_boxes)):
        box1 = combined_boxes[i]
        for j in range(len(combined_boxes)):
            box2 = combined_boxes[j]
            overlap = check_overlap(box1, box2)
            if overlap:
                box1 = combine_boxes(box1,box2)
        
        combined_boxes_final.append(box1)
        
        
    # remove duplicates from combined_boxes
    combined_boxes_final = remove_duplicates(combined_boxes_final)

    # remove boxes with small area
    combined_boxes_final = filter_boxes(combined_boxes_final)        
            
    return combined_boxes_final



def remove_duplicates(combined_boxes):

    new_boxes = set(tuple(x) for x in combined_boxes)
    combined_boxes = [list(x) for x in new_boxes]
    
    return combined_boxes




def filter_boxes(boxes):
    
    num_boxes = len(boxes)
    box_areas = []
    filtered_boxes = []
    
    for box in boxes:
        area = calculate_area(box)
        box_areas.append(area)
    
    avg_area = sum(box_areas)/num_boxes
    
    for box in boxes:
        area = calculate_area(box)
        if area > avg_area/2:
            filtered_boxes.append(box)
            
    return filtered_boxes



def check_overlap(box1, box2):
    
    b1x1 = box1[0]
    b1y1 = box1[1]
    b1x2 = box1[2]
    b1y2 = box1[3]
    
    b2x1 = box2[0]
    b2y1 = box2[1]
    b2x2 = box2[2]
    b2y2 = box2[3]
    
    x_overlap = False
    y_overlap = False   
    
    # check x coords for overlap
    if (b1x1 <= b2x1 <= b1x2 or b1x1 <= b2x2 <= b1x2) or (b2x1 <= b1x1 <= b2x2 or b2x1 <= b1x2 <= b2x2):
        x_overlap = True
        
    # check y coords for overlap
    if (b1y1 <= b2y1 <= b1y2 or b1y1 <= b2y2 <= b1y2) or (b2y1 <= b1y1 <= b2y2 or b2y1 <= b1y2 <= b2y2):
        y_overlap = True
            
    if x_overlap and y_overlap:
        overlap = True
    else:
        overlap = False
    
    return overlap



def remove_small(cnts):
    
    save_list = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 5000:
            save_list.append(c)
            
    cnts = save_list
    
    return cnts



def enlarge_boxes(x,y,w,h,offset,image_dims):
    
    img_h = image_dims[0]
    img_w = image_dims[1]
    
    x1 = max(0, x-offset)
    y1 = max(0, y-offset)
    x2 = min(x+w+offset, img_w)
    y2 = min(y+h+offset, img_h)
    
    
    return x1,y1,x2,y2



def calculate_area(box):
    
    x1,y1,x2,y2 = get_coords(box)
    
    area = abs(x2-x1)* abs(y2-y1)
    
    return area


def get_coords(box):
    
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    
    return x1,y1,x2,y2



def annotate_1r2(slide_number):
    
    from PIL import ImageDraw
    
    filename = "model_predictions_dict_" + str(slide_number) + "_filtered.pickle" 
        
    with open(cg.SAVED_DATABASE_DIR + filename, 'rb') as handle:
        filtered_predictions = pickle.load(handle)
    
    t = time.time()
    
    # get the current slide to annotated
    png_slide_path = utils.get_png_slide_path(slide_number)
    
    # make new empty image to place red 1r2 squares
    image = Image.open(png_slide_path)
    image_dims = image.size
    image = Image.new(mode='RGBA', size=(image_dims))
    
    draw = ImageDraw.Draw(image, 'RGBA')
    
    # Just get the filename of the current png file
    png_slide_name = utils.get_png_slide_name(slide_number)
    
    # get dimensions from original slide from png filename
    large_w, large_h, small_w, small_h = utils.parse_dimensions_from_image_filename(png_slide_name)
    
    # print("Original Dimensions = ", large_w, large_h, small_w, small_h)
    
    # ammount to offset the annotation to make the box centered
    # offset =  cg.PATCH_SIZE/2/cg.SCALE_FACTOR
    offset =  cg.ANNOTATION_SIZE/2/cg.SCALE_FACTOR

    
    counter = 1
    
    ## FIND COORDS OF THE CENTERS OF PATCHES TO ANNOTATE AND DRAW BOX
    for key, value in filtered_predictions.items():
        value = np.argmax(value)
        if (value == 1):
            color = (255, 0, 0, 255)
            patch = key
            box_size = 1.5
            
            # Get patch coordinates from filename
            large_x, large_y = utils.get_coords_from_name(key)
            
            # convert from large to small coordinates
            small_x, small_y = utils.large_to_small_coords(large_w, large_h, small_w, small_h, large_x, large_y)

            # Find coordinates of center of patch
            patch_center = [small_x + offset, small_y + offset]
            
            # define box top left / bottom right position
            # Draw color coded box in center of patch
            
            top_left = (patch_center[0] - box_size, patch_center[1] - box_size)
            bottom_right = (patch_center[0] + box_size, patch_center[1] + box_size)
    
            draw.rectangle([top_left, bottom_right], fill = color, outline = None)
            counter += 1

        
    save_dir = annotated_1r2_dir

    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    save_path = save_dir + str(slide_number) + "_1r2.png" 

    # Convert to RGB
    image = image.convert(mode = 'RGB')
    
    image.save(save_path, 'PNG')
    # print(f"Total time for annotating slide {time.time()-t}")
    
    # Clean Up    
    del draw



def segment_image(slide_number):

    import shutil

    # the image with only 1r2 annotations made above
    image_1r2 = annotated_1r2_dir +  str(slide_number) + "_1r2.png"
    image_1r2 = cv2.imread(image_1r2)
    
    # get the original filtered image for segementation
    images = os.listdir(filtered_image_dir) 
    for image in images:
        if image.split(".")[0].split("-")[-1] == "filtered" and image.split(".")[0].split("-")[0] == str(slide_number):
            filtered_image = image
            # print("Found image ", filtered_image)

    
    # segment tissue on the original filtered image
    image = cv2.imread(filtered_image_dir + filtered_image)
    image_copy = image.copy()
    image_orig = image.copy()


    image_dims = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 0, 255, 1)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)

    cnts, h = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # remove small contours
    cnts = remove_small(cnts)

    # Iterate thorugh contours and filter for ROI
    bounding_boxes = []

    # offset = amount to enlarge the bounding boxes
    offset = 90

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        # enlarge boxes and make sure bounding box doesnt go off the image
        x1,y1,x2,y2 = enlarge_boxes(x,y,w,h,offset,image_dims)


        cv2.rectangle(image_copy, (x1,  y1), (x2, y2), (36,255,12), 2)

        # save for combining later
        bounding_boxes.append([x1, y1, x2, y2])


    # save a copy of the raw boxes for analysis
    save_dir = bounding_boxes_dir

    if not os.path.isdir(bounding_boxes_dir): os.makedirs(bounding_boxes_dir)

    save_path = bounding_boxes_dir + str(slide_number) + "_orig_boxes.jpg"
    cv2.imwrite(save_path, image_copy)


    # Now combine the boxes for a tissue into one big box
    combined_boxes = analyze_boxes(bounding_boxes)

    # print("combined boxes = ", combined_boxes)


    #############################################################

    # save ROI on the 1R2 image
    path = roi_1r2_dir + str(slide_number)

    #delete old results
    if isdir(path): shutil.rmtree(path)

    #make directory
    utils.make_directory(path) 

    #############################################################


    roi_number = 0

    for box in combined_boxes:
        
        x1,y1,x2,y2 = get_coords(box)

        # save ROI on the 1R2 image
        utils.make_directory(roi_1r2_dir + str(slide_number)) 
        filename = roi_1r2_dir + str(slide_number) + "\\" + "roi_{}.png".format(roi_number)
        roi = image_1r2[y1:y2, x1:x2]
        cv2.imwrite(filename, roi)

        # save ROI on the filtered png image
        # Use later for further segmentation if needed (isolate the tissue in individual roi)
        utils.make_directory(roi_filter_dir + str(slide_number))
        filename = roi_filter_dir + str(slide_number) + "\\" + "roi_{}.png".format(roi_number)
        roi = image_orig[y1:y2, x1:x2]
        cv2.imwrite(filename, roi)


        roi_number += 1

        # Draw ROI on 1r2 image
        cv2.rectangle(image_1r2, (x1, y1), (x2, y2), (36,255,12), 2)

        # Draw ROI on original filtered image
        cv2.rectangle(image, (x1, y1), (x2, y2), (36,255,12), 2)

    
    # Save image with combined boxes to review  
    filename = combined_boxes_dir + str(slide_number) + "_combined_boxes.jpg"
    cv2.imwrite(filename, image)

    # Save image with combined boxes to review  
    filename = segmented_dir + "1R2_Only\\" + str(slide_number) + "_orig_boxes.jpg"
    cv2.imwrite(filename, image_1r2)



def analyze_segments(slide_number):
    

    path = roi_1r2_dir + str(slide_number) + "\\"
    all_roi = [file for file in os.listdir(path) if "dilated" not in file and "centroids" not in file]
            
    dilated_imgs = []

    for i in range(len(all_roi)):

        roi_name = all_roi[i].split(".")[0]
        roi = path + all_roi[i]
        image = Image.open(roi)
        image_gy = ImageOps.grayscale(image)
        image = np.array(image_gy)
        
        try:
            val = filters.threshold_otsu(image)
            patches = ndimage.binary_fill_holes(image > val)
            patches = ndimage.morphology.binary_dilation(patches, iterations=cg._1R2_DILATION_ITERS)

            dilated_imgs.append(patches)

            save_img = Image.fromarray(patches)
            save_path = path + roi_name + "_dilated.png"
            save_img.save(save_path)


        except:
            pass
    
    if len(dilated_imgs) > 0:

        padded_imgs = pad_images(dilated_imgs)

        composite_img = sum(padded_imgs)
        composite_img = composite_img > 0

        save_image = Image.fromarray(composite_img)
        save_path = path + "dilated_composite.png"
        save_image.save(save_path)

        groups = measure.label(composite_img)

        num_1r2 = groups.max()

    else:

        num_1r2 = 0


    return num_1r2



def pad_images(images):

    shapes = []

    for img in images: 
        shapes.append(list(img.shape))

    h = []
    w = []

    for shape in shapes:
        h.append(shape[0])
        w.append(shape[1])

    # create new image of desired size and color (black) for padding
    ww = max(w)
    hh = max(h)
    color = (0)

    padded_imgs = []

    for img in images: 
        result = np.full((hh,ww), color, dtype=np.uint8)
        ht, wd = img.shape
        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        result[yy:yy+ht, xx:xx+wd] = img
        padded_imgs.append(result)

    return padded_imgs



def main(slide_number):
    
    annotate_1r2(slide_number)
    segment_image(slide_number)

    num_1r2 = analyze_segments(slide_number)

    return num_1r2
    

    

if __name__ == '__main__': main()






