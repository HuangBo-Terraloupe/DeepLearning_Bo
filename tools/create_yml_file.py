from os import listdir
from os.path import exists
from PIL import Image
import numpy as np
import yaml
import os

import cv2
from random import shuffle


#########################################################################
#================== Set these ======================================

NAME = 'Regensburg roof dataset' #Name of the dataset (optional)
IMG_TYPE= "ortho"  # optional
IMG_CHANNELS = "rgb"
SOURCE_BITDEPTH = 8

CLASS_NAMES = ['Rooftop Area', 'Solar Thermal Panel',
               'Chimney/Ventilation Pipe', 'Roof Window']

GT_TYPE = "bbox" # or categorial / bbox
PREFIX = ''
IMAGE_DIR_NAME = '/home/huangbo/HuangBo_Projects/regensburg/ortho/'
MASKS_DIR_NAME = '/home/huangbo/Desktop/jsons_filter/'  #Only name, not full path
IMG_EXT = '.jpg'  #Images extensions
MASKS_EXT = '.json'  #Masks extensions
OUTFILE_NAME = "regensburg.yml"
TRAIN_RATIO = 1  # Training set ratio  Between 0-1 

#########################################################################

data = {"type": IMG_TYPE, "channels": IMG_CHANNELS, "source_bitdepth": SOURCE_BITDEPTH}
classes = [{"index": i, "name": cls} for i, cls in enumerate(CLASS_NAMES)]
ground_truth = {"type": GT_TYPE, "classes": classes}
                
# Get image and label lists                
images_dir = PREFIX + IMAGE_DIR_NAME + '/'
image_paths = []
masks_dir = PREFIX + MASKS_DIR_NAME + '/'
masks_paths = []

ids = listdir(masks_dir)
shuffle(ids)

for id in ids:

    image_path = IMAGE_DIR_NAME + '/' + id[:-4] + IMG_EXT
    image_paths.append(image_path)

    masks_path = MASKS_DIR_NAME + '/' + id[:-4] + MASKS_EXT
    masks_paths.append(masks_path)


total_images = len(image_paths)
n_train = int(total_images * TRAIN_RATIO)
n_val = total_images - n_train
train_list = image_paths[0:n_train]
val_list = image_paths[n_train: n_train + n_val]
mask_train_list = masks_paths[0:n_train]
mask_val_list = masks_paths[n_train: n_train + n_val]


train = {'images': train_list, 'labels': mask_train_list}
val = {'images': val_list, 'labels': mask_val_list}
    

dataset = {"name": NAME, 
           "prefix": PREFIX,
           "data": data, 
           "ground_truth": ground_truth, 
           'training': train, 
           'validation': val}

f = open( os.path.join(PREFIX, OUTFILE_NAME), 'wb')
yaml.dump(dataset, f, default_flow_style=False)
