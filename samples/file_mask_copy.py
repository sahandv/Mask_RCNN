#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:58:14 2018

@author: https://github.com/sahandv
"""
# =============================================================================
# If you have used the label generator code in this repository for generating
# ground truth data, you will notice that the data is not as good and pure as 
# you might need. Therefore, you can manually go through a manual elimination
# of bad detected frames. 
# This piece of code will get the purified (hand-picked) images from bbx or 
# folder, or any folder you say, then copy the corresponding xml and frames
# to another place to match the bounding boxes.
# Use this code at your own risk.
# =============================================================================


import os
#from os.path import isfile
from shutil import copyfile

image_directory = '/media/sahand/Archive Linux/Sahand/'

source_dir_img = '/media/sahand/Archive Linux/Data/CityIstanbul/Annotations_short_test/images/'
dest_dir_img = '/media/sahand/Archive Linux/Sahand/images/'

source_dir_xml = '/media/sahand/Archive Linux/Data/CityIstanbul/Annotations_short_test/annotations/'
dest_dir_xml = '/media/sahand/Archive Linux/Sahand/xml/'


all_files = []
for root, dirs, files in os.walk(image_directory):
    for file in files:
        if file.endswith('.jpg'):
            file_name = str.split(file,'.')[0]
            all_files.append(file_name)

for file_name in all_files:
    from_path_img = source_dir_img+file_name+'.jpg'
    to_path_img = dest_dir_img+file_name+'c'+'.jpg'
    
    #to_path_img = dest_dir_img+file_name+'.jpg'
    
    from_path_xml = source_dir_xml+file_name+'.xml'
    to_path_xml = dest_dir_xml+file_name+'c'+'.xml'  
    
    #to_path_xml = dest_dir_xml+file_name+'.xml'  
    
    copyfile(from_path_img, to_path_img)
    copyfile(from_path_xml, to_path_xml)