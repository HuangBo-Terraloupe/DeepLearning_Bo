import os
import yaml

from os import listdir
from random import shuffle

import cv2
import numpy as np

#########################################################################
#================== Set these ======================================

NAME = 'San Francisco here' #Name of the dataset (optional)

IMG_TYPE= "ortho"  # optional
IMG_CHANNELS = "rgb"
SOURCE_BITDEPTH = 8

CLASS_NAMES = ['roads']

GT_TYPE = "semseg" # semseg, detection, bbox

PREFIX = '/data/here/patches_osm/'

IMAGE_DIR_NAME = 'san_fancisco/images'
MASKS_DIR_NAME = 'san_fancisco/masks'  #Only name, not full path
IMG_EXT = '.png'  #Images extensions
MASKS_EXT = '.png'  #Masks extensions
OUTFILE_NAME = '/home/bo_huang/here/San_Francisco_road_segmentation/san_francisco_sampling.yml'
TRAIN_RATIO = [0.9, 0.1, 0.0]  # Training set ratio  Between 0-1

N_categories = 3
Add_Probs = True
category_diff_ratio = 0.03

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
ids.sort()
print('The number of training data before', len(ids))

ids = ids[0 : int(len(ids) / 2.0)]

print('The number of training data before', len(ids))

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

if Add_Probs is True:
    # probability sampling
    static_dic = {'negative': 0, 'positive': 0, 'median':0}
    for i, file in enumerate(masks_paths):
        mask_file = os.path.join(PREFIX, file)
        mask_file = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
        if np.count_nonzero(mask_file) == (mask_file.shape[0]*mask_file.shape[1]):
            static_dic['negative'] = static_dic['negative'] + 1
        elif np.count_nonzero(mask_file)/(mask_file.shape[0]*mask_file.shape[1]) > category_diff_ratio:
            static_dic['positive'] = static_dic['positive'] + 1
        else:
            static_dic['median'] = static_dic['median'] + 1

    prob_negative = float(static_dic['negative']) / (len(masks_paths) * N_categories)
    prob_positive = float(static_dic['positive']) / (len(masks_paths) * N_categories)
    prob_median = float(static_dic['median']) / (len(masks_paths) * N_categories)

    print('Statistic Info:')
    print(static_dic)
    print('positive, median, negative', prob_positive, prob_median, prob_negative)

    prob_list = []

    for i, file in enumerate(masks_paths):
        mask_file = os.path.join(PREFIX, file)
        mask_file = cv2.imread(mask_file, cv2.IMREAD_ANYDEPTH)
        if np.count_nonzero(mask_file) == (mask_file.shape[0]*mask_file.shape[1]):
            prob_list.append(prob_negative)
        elif np.count_nonzero(mask_file)/(mask_file.shape[0]*mask_file.shape[1]) > category_diff_ratio:
            prob_list.append(prob_positive)
        else:
            prob_list.append(prob_median)


    probs_train_list = prob_list[0:n_train]
    probs_val_list = prob_list[n_train: n_train + n_val]
    probs_test_list = prob_list[n_train + n_val: n_train + n_val + n_test]

    train = {'images': train_list, 'labels': mask_train_list, 'probability':probs_train_list}
    val = {'images': val_list, 'labels': mask_val_list, 'probability':probs_val_list}
    test = {'images': test_list, 'labels': mask_test_list, 'probability':probs_test_list}

else:
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

f = open(os.path.join(PREFIX, OUTFILE_NAME), 'w')
yaml.dump(dataset, f, default_flow_style=False)
