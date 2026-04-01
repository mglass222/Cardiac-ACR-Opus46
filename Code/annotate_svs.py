#!/usr/bin/env python
# coding: utf-8


import xml.etree.ElementTree as ET
import numpy as np

import os
import time
from os import listdir
from os.path import isfile, isdir, join
from pathlib import Path

import re
import pickle
import numpy as np

import cardiac_globals as cg
import cardiac_utils as utils



def annotate_slide(slide_number):
    
    extracted_slidename = get_extracted_slide_name(slide_number)
    
    xmlfilename = cg.TEST_SLIDE_ANNOTATIONS_DIR + str(slide_number) + ".xml"

    utils.make_directory(cg.TEST_SLIDE_ANNOTATIONS_DIR)
    
    # make new xml file
    initilialize_xml_file(xmlfilename)
    
    tree = ET.parse(xmlfilename)
    root = tree.getroot()
    
    all_dicts = load_diagnoses(slide_number)
    
    #Always starts at 1
    region_id = "1"

    for i in range(len(all_dicts)):
        current_dict = all_dicts[i]
        annotation_id = str(i)
        region_id = update_xml_file(root, current_dict, annotation_id, region_id)
        
    tree = ET.ElementTree(root)
    tree.write(xmlfilename)

    print(f"Writing XML file to: {xmlfilename}") 

    pretty_print(xmlfilename)


def pretty_print(xmlfilename):
    
    import xml.dom.minidom
    
    dom = xml.dom.minidom.parse(xmlfilename) 
    pretty_xml_as_string = dom.toprettyxml()
    
    with open(xmlfilename,'w') as outfile:
        outfile.writelines(pretty_xml_as_string)


def initilialize_xml_file(xmlfilename):
    
    root = ET.Element("Annotations")
    root.attrib['MicronsPerPixel'] = "0.252300"
    tree = ET.ElementTree(root)
    
    with open(xmlfilename, 'wb') as file:
        tree.write(file)
    


def load_diagnoses(slide_number):
    
    filename = "model_predictions_dict_" + str(slide_number) + "_filtered.pickle"
    
    with open(cg.SAVED_DATABASE_DIR + filename, 'rb') as handle:
        dx_dict = pickle.load(handle)
        
    # sort the dict by diagnosis lowest number to highest
    # dx_dict = {i:j for i,j in sorted(dx_dict.items(), key=lambda item:item[1])}
    
    _1R1A_dict = {key:value for key,value in dx_dict.items() if np.argmax(value) == 0}
    _1R2_dict = {key:value for key,value in dx_dict.items() if np.argmax(value) == 1}
    Healing_dict = {key:value for key,value in dx_dict.items() if np.argmax(value) == 2}
    Hemorrhage_dict = {key:value for key,value in dx_dict.items() if np.argmax(value) == 3}
    Normal_dict = {key:value for key,value in dx_dict.items() if np.argmax(value) == 4}
    Quilty_dict = {key:value for key,value in dx_dict.items() if np.argmax(value) == 5}
    
    ## Take N annotations for each dict
    num_1r1a = "all"
    num_1r2 = "all"
    num_heal = 0
    num_hemorrhage = "all"
    num_normal = 0
    num_quilty = "all"
    
    _1R1A_dict = random_sample(_1R1A_dict, num_1r1a)
    _1R2_dict = random_sample(_1R2_dict, num_1r2)
    healing_dict = random_sample(Healing_dict, num_heal)
    hemorrhage_dict = random_sample(Hemorrhage_dict, num_hemorrhage)
    normal_dict = random_sample(Normal_dict, num_normal)
    quilty_dict = random_sample(Quilty_dict, num_quilty)
    
    all_dicts = [_1R1A_dict, _1R2_dict, healing_dict, hemorrhage_dict, normal_dict, quilty_dict]
        
    return all_dicts



def random_sample(dx_dict, num_samples):
    
    import random
    
    return_dict = {}
    random_dict = {}
    
    if num_samples == "all":
        N = len(dx_dict)
    else:
        N = num_samples

    keys = list(dx_dict.keys())
    random.shuffle(keys)
    
    count = 0
    for key in keys:
        random_dict.update({key:dx_dict[key]}) 

    for key,value in random_dict.items():
        return_dict.update({key:value})
        count += 1
        if count >= N:
            break

    return return_dict



def update_xml_file(root, current_dict, annotation_id, region_id):
    
    dx_dict = {0:"1R1A", 1:"1R2", 2:"Healing", 3:"Hemorrhage", 4:"Normal", 5:"Quilty"}
    linecolor_dict = {"1R1A":"65535", "1R2":"255", "Healing":"16744448", "Hemorrhage":"0", "Normal":"65280", "Quilty":"16711808"}
    
    for key, value in current_dict.items():
        patchname = key
        dx = dx_dict.get(np.argmax(value))
        color = linecolor_dict.get(dx)

        annotations = root.findall(".//Annotation")

        # if no annotations have been made
        if len(annotations) == 0 or dx not in [annotations[i].get('Name') for i in range(len(annotations))]:
            # print("Initializing ", dx)
            annotation = initialize_annotation_type(root, dx, color, annotation_id)
        else:
            for j in range(len(annotations)):
                if annotations[j].get('Name') == dx:
                    annotation = annotations[j]
            
        annotation = add_region(annotation, dx, region_id, patchname)
        region_id  = str(int(region_id) + 1)
                
    return region_id     



def initialize_annotation_type(root, dx, color, annotation_id):
    
    
    # add entry for the first entry in the xml file (1r1a first since dict is sorted)
    FirstAnnotation = ET.SubElement(root, "Annotation")
    
    FirstAnnotation.attrib["Id"] = annotation_id
    FirstAnnotation.attrib['Name'] = dx
    FirstAnnotation.attrib['ReadOnly'] = "0"
    FirstAnnotation.attrib['NameReadOnly'] = "0"
    FirstAnnotation.attrib['LineColorReadOnly'] = "0"
    FirstAnnotation.attrib['Incremental'] = "0"
    FirstAnnotation.attrib['Type'] = "4"
    FirstAnnotation.attrib['Incremental'] = "0"
    FirstAnnotation.attrib['LineColor'] = color
    FirstAnnotation.attrib['Visible'] = "1"
    FirstAnnotation.attrib['Selected'] = "0"
    FirstAnnotation.attrib['Incremental'] = "0" 
    FirstAnnotation.attrib['MarkupImagePath'] = ""
    FirstAnnotation.attrib['MacroName'] = ""
    
    Attributes = ET.SubElement(FirstAnnotation, "Attributes")
    Attribute = ET.SubElement(Attributes, "Attribute")
    
    Attribute.attrib['Name'] = dx
    Attribute.attrib['Id'] = str(int(annotation_id) - 1) # starts with zero instead of one
    Attribute.attrib['Value'] = "0"

    # All Annotation types have this as thier first entry
    Regions = ET.SubElement(FirstAnnotation, "Regions")
    RegionsAttribs = ET.SubElement(Regions, "RegionAttributeHeaders")
    
    AttributeHeader = ET.SubElement(RegionsAttribs, "AttributeHeader")
    AttributeHeader.attrib['Id'] = "9999"
    AttributeHeader.attrib['Name'] = "Region"
    AttributeHeader.attrib['ColumnWidth'] = "-1"

    AttributeHeader = ET.SubElement(RegionsAttribs, "AttributeHeader")
    AttributeHeader.attrib['Id'] = "9997"
    AttributeHeader.attrib['Name'] = "Length"
    AttributeHeader.attrib['ColumnWidth'] = "-1"

    AttributeHeader = ET.SubElement(RegionsAttribs, "AttributeHeader")
    AttributeHeader.attrib['Id'] = "9996"
    AttributeHeader.attrib['Name'] = "Area"
    AttributeHeader.attrib['ColumnWidth'] = "-1"

    AttributeHeader = ET.SubElement(RegionsAttribs, "AttributeHeader")
    AttributeHeader.attrib['Id'] = "9998"
    AttributeHeader.attrib['Name'] = "Text"
    AttributeHeader.attrib['ColumnWidth'] = "-1"

    AttributeHeader = ET.SubElement(RegionsAttribs, "AttributeHeader")
    AttributeHeader.attrib['Id'] = "1"
    AttributeHeader.attrib['Name'] = "Description"
    AttributeHeader.attrib['ColumnWidth'] = "-1"
    
    return FirstAnnotation
    


def add_region(annotation, dx, region_id, patchname):
    
    Regions = annotation.find('.//Regions')
    Region = ET.SubElement(Regions, "Region")
    
    Region.attrib['Id'] = region_id
    Region.attrib['Type'] = "1"
    Region.attrib['Zoom'] = "1"
    Region.attrib['Selected'] = "0"
    Region.attrib['ImageLocation'] = ""
    Region.attrib['ImageFocus'] = "1"
    Region.attrib['Length'] = "896.0"
    Region.attrib['Area'] = "50176.0"
    Region.attrib['LengthMicrons'] = "226.1"
    Region.attrib['AreaMicrons'] = "3194"
    Region.attrib['Text'] = ""
    Region.attrib['NegativeROA'] = "0"
    Region.attrib['InputRegionId'] = "0"
    Region.attrib['Analyze'] = "1"
    Region.attrib['DisplayId'] = region_id
    
    # leave blank
    Attributes = ET.SubElement(Region, "Attributes")
    
    # Each set of vertices has a "vertices" and 4 "vertex-es"
    Vertices = ET.SubElement(Region, "Vertices")

    # get vertices from filename
    small_x, small_y, large_x, large_y = get_coords(patchname)
    
    # Add first Vertex
    Vertex = ET.SubElement(Vertices, "Vertex")
    Vertex.attrib['X'] = str(small_x)
    Vertex.attrib['Y'] = str(small_y)
    
    # Add Second Vertex
    Vertex = ET.SubElement(Vertices, "Vertex")
    Vertex.attrib['X'] = str(large_x)
    Vertex.attrib['Y'] = str(small_y)
    
    # Add Third Vertex
    Vertex = ET.SubElement(Vertices, "Vertex")
    Vertex.attrib['X'] = str(large_x)
    Vertex.attrib['Y'] = str(large_y)
    
    # Add fourth Vertex
    Vertex = ET.SubElement(Vertices, "Vertex")
    Vertex.attrib['X'] = str(small_x)
    Vertex.attrib['Y'] = str(large_y)
    
    
    return annotation



def get_coords(patchname):
    
    """
    Order of vertex entries:
    vertex = (small_x, small_y)
    vertex = (large_x, small_y)
    vertex = (small_x, large_y)
    vertex = large_x, large_y)
    """
    x, y = get_coords_from_name(patchname)
    
    # original size of patches
    patch_size = 224
    
    # make slightly smaller than 224 so borders dont overlap
    box_size = 215
    
    # offset to center of patch
    offset = patch_size/2 
            
    # Find coordinates of center of patch
    patch_center = [x + offset, y + offset]

    small_x = patch_center[0] - box_size/2
    small_y = patch_center[1] - box_size/2
    large_x = patch_center[0] + box_size/2 
    large_y = patch_center[1] + box_size/2
    
    return small_x, small_y, large_x, large_y



def get_extracted_slide_name(slide_number):
    files = listdir(cg.PNG_SLIDE_DIR)

    return_file = None
    for file in files:
        if file.split(".")[0].split("-")[0] == str(slide_number):
            return_file = file
            
    return return_file



def get_coords_from_name(name):
    
    # Get coordinates from filename
    m = re.match(".*-x([\d]*)-y([\d]*).*\..*", name)
    x_coord = int(m.group(1))
    y_coord = int(m.group(2))
    
    return x_coord, y_coord



def main(slide_number):
    
    # slides = [file.split(".")[0] for file in listdir(cg.TEST_SLIDE_DIR) if file.split(".")[1] == "svs"]

    annotate_slide(slide_number)
        

if __name__ == "__main__": main()







