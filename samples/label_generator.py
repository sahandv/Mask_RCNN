#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:58:21 2018

@author: https://github.com/sahandv
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# =============================================================================
# Configurations
# =============================================================================

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# =============================================================================
# Create Model and Load Trained Weights
# =============================================================================

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
#class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#               'bus', 'train', 'truck', 'boat', 'traffic light',
#               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#               'kite', 'baseball bat', 'baseball glove', 'skateboard',
#               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#               'teddy bear', 'hair drier', 'toothbrush']

class_names = ['BG', 'person', 'bike', 'car', 'bike', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# =============================================================================
# Run Object Detection
# =============================================================================

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image_path = os.path.join(IMAGE_DIR, random.choice(file_names))
image = skimage.io.imread('/home/sahand/Projects/Mask_RCNN/images/ist26.jpg')
image_cv = cv2.imread(image_path)
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
white_list = [1,2,3,4,6,8]
#r_whitelisted ={'class_ids': [],
#                'masks':[],
#                'rois':[],
#                'scores':[]}
#
#
#
#idx = 0
#white_list = [1,2,3,4,6,8]
#class_ids = []
#rois = []
#masks = []
#scores = []
#for i in r['class_ids']:
#    if i in white_list:
#        class_ids.append(r['class_ids'][idx])
#        masks.append(r['masks'][idx])
#        rois.append(r['rois'][idx])
#        scores.append(r['scores'][idx])
#    idx = idx + 1
#    
#r_whitelisted['class_ids']= class_ids
#r_whitelisted['masks']= masks
#r_whitelisted['rois']= rois
#r_whitelisted['scores']= scores
    
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names,show_mask=False,show_bbox=True,show_contours=False, white_list=white_list)

# =============================================================================
## TODO:
#1- read video
#2- extract frames
#3- iterate over frames
#4- perform detection (r = model.detect([image], verbose=1)) 
#5- define class whitelist for results
#6- write new visualization code based on opencv (get cue from ssd-keras repo)
#7- write parser for xml output (get cue from voc-maker)
#8- write frame, framw with box and xml for each frame to folder
# =============================================================================



















