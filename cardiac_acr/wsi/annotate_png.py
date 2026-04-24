# import project modules ###

import os
import pickle
import time

import numpy as np
from PIL import Image

from cardiac_acr import config as cg
from cardiac_acr.utils import cardiac_utils as utils



def annotate_png(slide_number, saved_database_dir, annotated_png_dir):

	from PIL import ImageDraw

	filename = "model_predictions_dict_" + str(slide_number) + "_filtered.pickle"

	with open(os.path.join(saved_database_dir, filename), 'rb') as handle:
		filtered_predictions = pickle.load(handle)
	
	t = time.time()
	
	# get the current slide to annotated
	png_slide_path = utils.get_png_slide_path(slide_number)
	
	# open image for annotation
	image = Image.open(png_slide_path)
	draw = ImageDraw.Draw(image, 'RGBA')
	
	# Just get the filename of the current png file
	png_slide_name = utils.get_png_slide_name(slide_number)
	
	# get dimensions from original slide from png filename
	large_w, large_h, small_w, small_h = utils.parse_dimensions_from_image_filename(png_slide_name)
	
	# print("Original Dimensions = ", large_w, large_h, small_w, small_h)
	
	# ammount to offset the annotation to make the box centered
	offset = cg.ANNOTATION_SIZE/2/cg.SCALE_FACTOR

	
	## FIND COORDS OF THE CENTERS OF PATCHES TO ANNOTATE AND DRAW BOX
	for key, value in filtered_predictions.items():
		value = np.argmax(value)
		color = get_color(value)
		if (color == None):
			pass
		else:
			patch = key

			box_size = 2

			# Get patch coordinates from filename
			large_x, large_y = utils.get_coords_from_name(key)

			
			# convert from large to small coordinates
			small_x, small_y = utils.large_to_small_coords(large_w, large_h, small_w, small_h, large_x, large_y)


			# Find coordinates of center of patch
			patch_center = [small_x + offset, small_y + offset]

			
			# define box top left / bottom right position
			top_left = (patch_center[0] - box_size, patch_center[1] - box_size)
			bottom_right = (patch_center[0] + box_size, patch_center[1] + box_size)

	
			# Draw color coded box in center of patch
			draw.rectangle([top_left, bottom_right], fill = color, outline = None)


		
	save_dir = os.path.join(annotated_png_dir, "CONFIDENCE " + str(cg.PREDICTION_THRESHOLD*100) + "%")

	utils.make_directory(save_dir)

	save_path = os.path.join(save_dir, str(slide_number) + ".png")
	
	print("Saving Annotated PNG to: ", save_path)
	
	# Convert to RGB
	image = image.convert(mode = 'RGB')
	
	image.save(save_path, 'PNG')
	# print(f"Total time for annotating slide {time.time()-t}")
	
	# Clean Up    
	del draw


def get_color(value):

	# 1R1A yellow
	if (value == 0): 
		color = (255,255,0,255)

	# 1R2 red    
	elif (value == 1): 
		color = (255,0,0,255)

	# Healing blue    
	elif (value == 2): 
		color = (0,0,255,255)

	# Hemorrhage black
	elif (value == 3):
		color = (0,0,0,255)

	# Normal green
	elif (value == 4): 
		color = (0,255,0,255)

	# Quilty purple
	elif (value == 5): 
		color = (158,0,255,255)

	else:
		color = None
		
	return color


def main(slide_number, saved_database_dir, annotated_png_dir):

	annotate_png(slide_number, saved_database_dir, annotated_png_dir)


if __name__ == "__main__": main()
