#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:58:21 2018

@author: https://github.com/sahandv
"""

import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv2
from lxml import etree as et
from imageio import get_reader
from imageio import imwrite
from imageio import get_writer


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


#file_name = 'ist26.jpg'
#image_path = os.path.join('/home/sahand/Projects/Mask_RCNN/images/', file_name)
video_path = '/media/sahand/Archive Linux/Data/CityIstanbul/20181118_165314.mp4'
output_dir = '/media/sahand/Archive Linux/Data/CityIstanbul/Annotations_20181118_165314/'
output_frame_prefix = '20181118_165314-'
#image = skimage.io.imread(image_path)
#orig_height, orig_width, channels = image_cv.shape
white_list = [1,2,3,4,6,8]

# =============================================================================
# Prepare directories
# =============================================================================

if not os.path.exists(output_dir):
    os.makedirs(os.path.join(output_dir, 'images'))
    os.makedirs(os.path.join(output_dir, 'images_bbox'))
    os.makedirs(os.path.join(output_dir, 'annotations'))


# =============================================================================
# Read Video
# =============================================================================
reader = get_reader(video_path)

for iframe, frame in enumerate(reader):
    print('processing frame ',iframe)
    start_time = time.time()
    xml_out_path =  output_dir+'annotations'+'/'+output_frame_prefix+str(iframe)+'.xml'
    img_out_path =  output_dir+'images'+'/'+output_frame_prefix+str(iframe)+'.jpg'
    img_bbx_out_path =  output_dir+'images_bbox'+'/'+output_frame_prefix+str(iframe)+'.jpg'
    
    orig_height, orig_width, channels = frame.shape
    # =============================================================================
    # Predict
    # =============================================================================
    
    # Run detection
    results = model.detect([frame], verbose=1)
    
    # Visualize results
    r = results[0]
    
    visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                                class_names,show_mask=False,show_bbox=True,
                                show_contours=False, white_list=white_list,
                                frame_save_path = img_bbx_out_path,
                                save_img = True,show_img = False)
    
    # =============================================================================
    # Parser XML
    # =============================================================================
    xml_annotation = et.Element('annotation')
    xml_folder = et.SubElement(xml_annotation, 'folder')
    xml_filename = et.SubElement(xml_annotation, 'filename')
    xml_path = et.SubElement(xml_annotation, 'path')
    xml_source = et.SubElement(xml_annotation, 'source')
    xml_database = et.SubElement(xml_source,'database')
    xml_size = et.SubElement(xml_annotation, 'size')
    xml_width = et.SubElement(xml_size,'width')
    xml_height = et.SubElement(xml_size,'height')
    xml_depth = et.SubElement(xml_size,'depth')
    xml_segmented = et.SubElement(xml_annotation, 'segmented')
    
    xml_folder.text = '' 
    xml_filename.text = str.split(img_out_path,os.sep)[-1]
    xml_path.text = img_out_path
    xml_database.text = 'Unknown'
    xml_width.text = str(int(orig_width))
    xml_height.text = str(int(orig_height))
    xml_depth.text = str(channels)                                               # For RGB/color images
    xml_segmented.text = '0'
    
    idx = 0
    for class_id in r['class_ids']:
        if class_id in white_list:
            label = class_names[class_id]
            xmax = 0
            xml_object = et.SubElement(xml_annotation, 'object')
            xml_name = et.SubElement(xml_object,'name')
            xml_pose = et.SubElement(xml_object,'pose')
            xml_truncated = et.SubElement(xml_object,'truncated')
            xml_difficult = et.SubElement(xml_object,'difficult')
            xml_bndbox = et.SubElement(xml_object,'bndbox')
            xml_xmin = et.SubElement(xml_bndbox,'xmin')
            xml_ymin = et.SubElement(xml_bndbox,'ymin')
            xml_xmax = et.SubElement(xml_bndbox,'xmax')
            xml_ymax = et.SubElement(xml_bndbox,'ymax')
            
            xml_name.text = label
            xml_pose.text = 'Unspecified'
            xml_truncated.text = '0'
            xml_difficult.text = '0'
                
            xmax = r['rois'][idx][3]
            ymax = r['rois'][idx][2]
            xmin = r['rois'][idx][1]
            ymin = r['rois'][idx][0]
            
            xml_xmin.text = str(int(xmin))
            xml_ymin.text = str(int(ymin))
            xml_xmax.text = str(int(xmax))
            xml_ymax.text = str(int(ymax))
            
        idx = idx+1
    
    mydata = et.tostring(xml_annotation, pretty_print=True)  
    with open(xml_out_path, "wb") as f1:
        f1.write(mydata)
    
    imwrite(img_out_path,frame)
    
    print('Frame processing time: '+str((time.time() - start_time))+'s')





