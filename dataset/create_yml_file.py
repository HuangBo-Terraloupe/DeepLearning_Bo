import os
import yaml

from os import listdir
from random import shuffle


#########################################################################
#================== Set these ======================================

NAME = 'Hollywood' #Name of the dataset (optional)

IMG_TYPE= "ortho"  # optional
IMG_CHANNELS = "rgb"
SOURCE_BITDEPTH = 8

CLASS_NAMES = ['object']

GT_TYPE = "bbox" # or categorial / bbox
PREFIX = '/home/ga58zak/Hollywood_dataset/'

IMAGE_DIR_NAME = 'images'
MASKS_DIR_NAME = 'annotations'  #Only name, not full path
IMG_EXT = 'jpg'  #Images extensions
MASKS_EXT = 'json'  #Masks extensions
OUTFILE_NAME = '/home/ga58zak/Hollywood_dataset/hollywood.yml'
TRAIN_RATIO = [0.95, 0.05, 0.0]  # Training set ratio  Between 0-1


#########################################################################

# #########################################################################
# #================== Set these ======================================
#
# NAME = 'harvey roof dataset' #Name of the dataset (optional)
# IMG_TYPE= "ortho"  # optional
# IMG_CHANNELS = "rgb"
# SOURCE_BITDEPTH = 8
#
# CLASS_NAMES = ['Rooftop Area',
#                'Panel',
#                'Chimney/Ventilation Pipe',
#                'Roof Window',
#                ]
#
# GT_TYPE = "bbox" # or categorial / bbox
# PREFIX = '/home/huangbo/HuangBo_Projects/regensburg/'
# IMAGE_DIR_NAME = 'images'
# MASKS_DIR_NAME = 'json_merge'  #Only name, not full path
# IMG_EXT = 'jpg'  #Images extensions
# MASKS_EXT = 'json'  #Masks extensions
# OUTFILE_NAME = "/home/huangbo/HuangBo_Projects/regensburg/model_object/regensburg.yml"
# TRAIN_RATIO = [0.8, 0.1, 0.1]  # Training set ratio  Between 0-1
#
# #########################################################################

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
n_train = int(total_images * TRAIN_RATIO[0])
n_val = int(total_images * TRAIN_RATIO[1])
n_test = int(total_images * TRAIN_RATIO[2])


train_list = image_paths[0:n_train]
val_list = image_paths[n_train: n_train + n_val]
test_list = image_paths[n_train + n_val: n_train + n_val + n_test]



mask_train_list = masks_paths[0:n_train]
mask_val_list = masks_paths[n_train: n_train + n_val]
mask_test_list = masks_paths[n_train + n_val: n_train + n_val + n_test]


train = {'images': train_list, 'labels': mask_train_list}
val = {'images': val_list, 'labels': mask_val_list}
test = {'images': test_list, 'labels': mask_test_list}
    

dataset = {"name": NAME, 
           "prefix": PREFIX,
           "data": data, 
           "ground_truth": ground_truth, 
           'training': train, 
           'validation': val,
           'testing': test
           }

f = open(os.path.join(PREFIX, OUTFILE_NAME), 'wb')
yaml.dump(dataset, f, default_flow_style=False)
