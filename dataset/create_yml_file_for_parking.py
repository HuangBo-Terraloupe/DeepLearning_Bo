import os
import yaml

from os import listdir
from random import shuffle
from glob import glob

#########################################################################
# ================== Set these ======================================

NAME = 'Continental Parking'  # Name of the dataset (optional)

IMG_TYPE = "ortho"  # optional
IMG_CHANNELS = "rgb"
SOURCE_BITDEPTH = 8

CLASS_NAMES = ['background', 'parking']

GT_TYPE = "semseg"  # semseg, detection, bbox

PREFIX = '/mnt/disks/conti-parking/germany_v2/'

IMAGE_DIR_NAME = 'images'
MASKS_DIR_NAME = 'masks'  # Only name, not full path
IMG_EXT = '.tif'  # Images extensions
MASKS_EXT = '.tif'  # Masks extensions
OUTFILE_NAME = '/home/bo_huang/parking.yml'


data = {"type": IMG_TYPE, "channels": IMG_CHANNELS, "source_bitdepth": SOURCE_BITDEPTH}
classes = [{"index": i, "name": cls} for i, cls in enumerate(CLASS_NAMES)]
ground_truth = {"type": GT_TYPE, "classes": classes}

# get images and labels
parking_images = glob(PREFIX + IMAGE_DIR_NAME + '/' + 'parking*' + IMG_EXT)
background_images = glob(PREFIX + IMAGE_DIR_NAME + '/' + 'background*' + IMG_EXT)
highway_images = glob(PREFIX + IMAGE_DIR_NAME + '/' + 'highway*' + IMG_EXT)

parking_images = [os.path.split(f)[-1] for f in parking_images]
background_images = [os.path.split(f)[-1] for f in background_images]
highway_images = [os.path.split(f)[-1] for f in highway_images]

shuffle(parking_images)
shuffle(background_images)
shuffle(highway_images)

train_list = parking_images[15000:] + background_images[5000:] + highway_images[5000:]
val_list = parking_images[0:15000] + background_images[0:5000] + highway_images[0:5000]
test_list = []

mask_train_list = parking_images[15000:] + background_images[5000:] + highway_images[5000:]
mask_val_list = parking_images[0:15000] + background_images[0:5000] + highway_images[0:5000]
mask_test_list = []

train_list = ['images/' + f for f in train_list]
val_list = ['images/' + f for f in val_list]


mask_train_list = ['masks/' + f for f in mask_train_list]
mask_val_list = ['masks/' + f for f in mask_val_list]


probs_parking = 0.5 / len(parking_images[15000:])
probs_background = 0.1 / len(background_images[5000:])
probs_highway = 0.4 / len(highway_images[5000:])

probs_train_list = [probs_parking] * len(parking_images[15000:]) + \
                   [probs_background] * len(background_images[5000:]) + [probs_highway] * len(highway_images[5000:])

train = {'images': train_list, 'labels': mask_train_list, 'probability': probs_train_list}
val = {'images': val_list, 'labels': mask_val_list, 'probability': []}
test = {'images': test_list, 'labels': mask_test_list, 'probability': []}

dataset = {"name": NAME,
           "prefix": PREFIX,
           "data": data,
           "ground_truth": ground_truth,
           'training': train,
           'validation': val,
           'testing': test
           }

f = open(os.path.join(PREFIX, OUTFILE_NAME), 'w')
yaml.dump(dataset, f, default_flow_style=False)
