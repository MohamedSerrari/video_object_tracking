import os 
import argparse
import cv2
from random import randint
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path


import math
from scipy.stats import mode
from sklearn.cluster import KMeans

# class names dictionnary 
class_names = {0 : 'Bowl', 1 : 'CanOfCocaCola', 2 : 'MilkBottle', 3 : 'Rice', 4 : 'Sugar'}
class_indexes = {'Bowl': 0, 'CanOfCocaCola' : 1, 'MilkBottle' : 2, 'Rice' : 3, 'Sugar' : 4}


def get_candidate_patches(frame, candidate_boxes):
    patches = []
    for box in candidate_boxes:
        x, y, w, h = [int(t) for t in box]
        crop = frame[y:y + h, x:x + w] # crop
        patches.append(cv2.resize(crop, (227, 227))) # resize

    return np.stack(patches, axis=0) # return stacked patches


def read_annotation_file(file_path):
    # open txt file of boxes 

    class_name = Path(file_path).name.split('Place')[0]

    print('class_name:', class_name)
    label_index = class_indexes.get(class_name)

    assert label_index is not None, ValueError(f'Could not get label index from file {file_path}')
    
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        boxes = { idx: [int(x) for x in row.split()[2:6]] for idx, row in enumerate(lines) if len(row.split()) == 6}
    # print()

    return label_index, boxes

def get_candidate_boxes(box, steps, aspect_ratios, im_width, im_height):

    x, y, w, h = box

    boxes = []
    
    for ratio in aspect_ratios:
        for step in steps: 
            boxes += [[x + step*w , y , ratio[0] * w, ratio[1] * h],
                      [x - step*w , y , ratio[0] * w, ratio[1] * h],
                      [x , y + step*h, ratio[0] * w, ratio[1] * h],
                      [x , y - step*h, ratio[0] * w, ratio[1] * h]]

    boxes = np.array(boxes).astype(int)

    return clip_boxes(boxes, im_width, im_height)


def clip_boxes(boxes, im_width, im_height):

    xyxy_boxes = cxcywh2xyxy(boxes)

    # Clamp boxes outside of the bounding box
    cliped_boxes = np.zeros_like(boxes)
    cliped_boxes[:,0] =  np.clip(xyxy_boxes[:, 0], a_min=0, a_max=im_width-1)  # x
    cliped_boxes[:,1] =  np.clip(xyxy_boxes[:, 1], a_min=0, a_max=im_height-1) # y
    cliped_boxes[:,2] =  np.clip(xyxy_boxes[:, 2], a_min=0, a_max=im_width-1)  # x + w
    cliped_boxes[:,3] =  np.clip(xyxy_boxes[:, 3], a_min=0, a_max=im_height-1) # y + h

    new_xywh_boxes = xyxy2cxcywh(cliped_boxes)
    
    # a box is unclamped if all its new coords are equal to the original unclamped coords
    diff_boxes = abs(new_xywh_boxes - boxes) <= 20 

    # boolean indices of boxes that the crop removed 40 pixels in each dimension at most
    inside = np.array([all(b) for b in diff_boxes])

    return boxes[inside]

def xyxy2cxcywh(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
    out[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    out[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return out

def cxcywh2xywh(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x left
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y top
    out[:, 2] = boxes[:, 2]  # width
    out[:, 3] = boxes[:, 3]  # height
    return out

def cxcywh2xyxy(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
    return out

def xyxy2xywh(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] # x left
    out[:, 1] = boxes[:, 1] # y top
    out[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    out[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return out

def xywh2xyxy(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0]  # top left x
    out[:, 1] = boxes[:, 1]  # top left y
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
    return out

def xywh2cxcywh(boxes):
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
    out[:, 2] = boxes[:, 2]  
    out[:, 3] = boxes[:, 3]
    return out

# function that returns squarred bounding boxe based on the original bounding box
def squared_boxes(x,y,w,h):
    size_diff = abs((w-h)//2)
    if w >= h :
        h = w 
        y = y - size_diff # center the box
    else : 
        w = h
        x = x - size_diff # center the box
    return (x,y,w,h)

# This function allows to calculate optical flow trajectories (Don't remember where I actually found the source code)
# The code also allows to specify step value. The greater the value the more sparse the calculation and visualisation
def calc_angl_n_transl(img, flow, step=8):
    
    '''
    input:
        - img - numpy array - image
        - flow - numpy array - optical flow
        - step - int - measurement of sparsity
    output:
        - angles - numpy array - array of angles of optical flow lines to the x-axis
        - translation - numpy array - array of length values for optical flow lines
        - lines - list - list of actual optical flow lines (where each line represents a trajectory of 
        a particular point in the image)
    '''

    angles = []
    translation = []

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    for (x1, y1), (x2, y2) in lines:
        angle = math.atan2(- int(y2) + int(y1), int(x2) - int(x1)) * 180.0 / np.pi
        length = math.hypot(int(x2) - int(x1), - int(y2) + int(y1))
        translation.append(length)
        angles.append(angle)
    
    return np.array(angles), np.array(translation), lines

# function that analyses optical flow information
def estimate_motion(angles, translation):
    
    '''
    Input:
        - angles - numpy array - array of angles of optical flow lines to the x-axis
        - translation - numpy array - array of length values for optical flow lines
    Output:
        - ang_mode - float - mode of angles of trajectories. can be used to determine the direction of movement
        - transl_mode - float - mode of translation values 
        - ratio - float - shows how different values of translation are across a pair of frames. allows to 
        conclude about the type of movement
        - steady - bool - show if there is almost no movement on the video at the moment
    '''
    
    # Get indices of nonzero opical flow values. We'll use just them
    nonzero = np.where(translation > 0)
    
    # Whether non-zero value is close to zero or not. Should be set as a thershold
    steady = np.mean(translation) < 0.5
    
    translation = translation[nonzero]
    transl_mode = mode(translation)[0][0]
    
    angles = angles[nonzero]
    ang_mode = mode(angles)[0][0]
    
    # cutt off twenty percent of the sorted list from both sides to get rid off outliers
    ten_percent = len(translation) // 10
    translations = sorted(translation)
    translations = translations[ten_percent: len(translations) - ten_percent]

    # cluster optical flow values and find out how different these cluster are
    # big difference (i.e. big ratio value) corresponds to panning, otherwise - trucking
    inliers = [tuple([inlier]) for inlier in translations]
    k_means = KMeans(n_clusters=3, random_state=0).fit(inliers)
    centers = sorted(k_means.cluster_centers_)
    ratio = centers[0] / centers[-1]
    
    return ang_mode, transl_mode, ratio, steady