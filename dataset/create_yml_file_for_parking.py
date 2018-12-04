import os
import yaml

from random import shuffle
from glob import glob


NAME = 'Continental Parking'  # Name of the dataset (optional)

IMG_TYPE = "ortho"  # optional
IMG_CHANNELS = "rgb"
SOURCE_BITDEPTH = 8

CLASS_NAMES = ['background', 'parking']

GT_TYPE = "semseg"  # semseg, detection, bbox

PREFIX = '/mnt/disks/conti-parking/germany_v2/'

IMAGE_DIR_NAME = 'images'
MASKS_DIR_NAME = 'annotations'  # Only name, not full path
IMG_EXT = '.tif'  # Images extensions
MASKS_EXT = '.json'  # Masks extensions
OUTFILE_NAME = '/home/bo_huang/continental_detection/parking_all_v1.yml'


data = {"type": IMG_TYPE, "channels": IMG_CHANNELS, "source_bitdepth": SOURCE_BITDEPTH}
classes = [{"index": i, "name": cls} for i, cls in enumerate(CLASS_NAMES)]
ground_truth = {"type": GT_TYPE, "classes": classes}

# get images and labels
parking_images = glob(PREFIX + MASKS_DIR_NAME + '/' + 'parking*' + MASKS_EXT)
background_images = glob(PREFIX + MASKS_DIR_NAME + '/' + 'background*' + MASKS_EXT)
highway_images = glob(PREFIX + MASKS_DIR_NAME + '/' + 'highway*' + MASKS_EXT)

parking_images = [os.path.split(f)[-1] for f in parking_images]
background_images = [os.path.split(f)[-1] for f in background_images]
highway_images = [os.path.split(f)[-1] for f in highway_images]


parking_images = [f.split('.')[0] for f in parking_images]
background_images = [f.split('.')[0] for f in background_images]
highway_images = [f.split('.')[0] for f in highway_images]

shuffle(parking_images)
shuffle(background_images)
shuffle(highway_images)

print('parking', len(parking_images), 'bg', len(background_images), 'highway', len(highway_images))


train_list = parking_images[5000:] + highway_images[2500:] + background_images[2500:]
val_list = parking_images[0:5000] + highway_images[0:2500] + background_images[0:2500]
test_list = []

mask_train_list = parking_images[5000:] + highway_images[2500:] + background_images[2500:]
mask_val_list = parking_images[0:5000] + highway_images[0:2500] + background_images[0:2500]
mask_test_list = []

train_list = [f + IMG_EXT for f in train_list]
val_list = [f + IMG_EXT for f in val_list]

mask_train_list = [f + MASKS_EXT for f in mask_train_list]
mask_val_list = [f + MASKS_EXT for f in mask_val_list]

train_list = ['images/' + f for f in train_list]
val_list = ['images/' + f for f in val_list]

mask_train_list = ['annotations/' + f for f in mask_train_list]
mask_val_list = ['annotations/' + f for f in mask_val_list]


train = {'images': train_list, 'labels': mask_train_list}
val = {'images': val_list, 'labels': mask_val_list}
test = {'images': [], 'labels': []}

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
