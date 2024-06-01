#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    
    tolerance = 20
    
    # check x coords for overlap
    if b1x1 <= b2x1 <= b1x2:
        amount = abs(b1x2-b2x1)
        if amount > tolerance:
            x_overlap = True
    
    elif b1x1 <= b2x2 <= b1x2:
        amount = abs(b1x1-b2x2)
        if amount > tolerance:
            x_overlap = True
    
    # check y coords for overlap
    if b1y1 <= b2y1 <= b1y2:
        amount = abs(b1y2-b2y1)
        if amount > tolerance:
            y_overlap = True
        
    elif b1y1 <= b2y2 <= b1y2:
        amount = abs(b1y1-b2y2)
        if amount > tolerance:
            y_overlap = True
            
    if x_overlap and y_overlap:
        overlap = True
    else:
        overlap = False
    
    return overlap


# In[ ]:


def calculate_area(box):
    
    x1,y1,x2,y2 = get_coords(box)
    
    area = abs(x2-x1)* abs(y2-y1)
    
    return area


# In[ ]:


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


# In[ ]:


def remove_duplicates(combined_boxes):

    new_boxes = set(tuple(x) for x in combined_boxes)
    combined_boxes = [list(x) for x in new_boxes]
    
    return combined_boxes


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def get_coords(box):
    
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    
    return x1,y1,x2,y2

