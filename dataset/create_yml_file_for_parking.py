import os
import yaml

from random import shuffle
from glob import glob

# configuration
NAME = 'Continental Parking'  # Name of the dataset (optional)

IMG_TYPE = "ortho"  # optional
IMG_CHANNELS = "rgb"
SOURCE_BITDEPTH = 8

CLASS_NAMES = ['background', 'parking']

GT_TYPE = "semseg"  # semseg, detection, bbox

PREFIX = '/home/bo_huang/training_data/'

IMAGE_DIR_NAME = 'images'
MASKS_DIR_NAME = 'masks'  # Only name, not full path
IMG_EXT = '.jpg'  # Images extensions
MASKS_EXT = '.tif'  # Masks extensions
OUTFILE_NAME = '/home/bo_huang/training_data/parking_manuel_without_hannover.yml'
TRAIN_RATIO = [0.8, 0.1, 0.1]  # Training set ratio  Between 0-1

data = {"type": IMG_TYPE, "channels": IMG_CHANNELS, "source_bitdepth": SOURCE_BITDEPTH}
classes = [{"index": i, "name": cls} for i, cls in enumerate(CLASS_NAMES)]
ground_truth = {"type": GT_TYPE, "classes": classes}

########################################################################################################################

# get images
images = glob(PREFIX + IMAGE_DIR_NAME + '/*' + IMG_EXT)
images = [os.path.split(f)[-1] for f in images]
images_names = [f.split('.')[0] for f in images]

# get masks
masks = glob(PREFIX + MASKS_DIR_NAME + '/*' + MASKS_EXT)
masks = [os.path.split(f)[-1] for f in masks]
masks_names = [f.split('.')[0] for f in masks]

# get files both in images and masks
valid_files = list(set(images_names).intersection(masks_names))
print('found valid number of training samples:', len(valid_files))

# shuffle the training data
shuffle(valid_files)

# get ratio to split training, validation and test set
n_train = int(len(valid_files) * TRAIN_RATIO[0])
n_val = int(len(valid_files) * TRAIN_RATIO[1])
n_test = int(len(valid_files) * TRAIN_RATIO[2])

image_train_list = valid_files[0:n_train]
image_val_list = valid_files[n_train: n_train + n_val]
image_test_list = valid_files[n_train + n_val: n_train + n_val + n_test]

mask_train_list = valid_files[0:n_train]
mask_val_list = valid_files[n_train: n_train + n_val]
mask_test_list = valid_files[n_train + n_val: n_train + n_val + n_test]

image_train_list = [f + IMG_EXT for f in image_train_list]
image_val_list = [f + IMG_EXT for f in image_val_list]
image_test_list = [f + IMG_EXT for f in image_test_list]

mask_train_list = [f + MASKS_EXT for f in mask_train_list]
mask_val_list = [f + MASKS_EXT for f in mask_val_list]
mask_test_list = [f + MASKS_EXT for f in mask_test_list]

image_train_list = ['images/' + f for f in image_train_list]
image_val_list = ['images/' + f for f in image_val_list]
image_test_list = ['images/' + f for f in image_test_list]

mask_train_list = ['masks/' + f for f in mask_train_list]
mask_val_list = ['masks/' + f for f in mask_val_list]
mask_test_list = ['masks/' + f for f in mask_test_list]


train = {'images': image_train_list, 'labels': mask_train_list}
val = {'images': image_val_list, 'labels': mask_val_list}
test = {'images': image_test_list, 'labels': mask_test_list}

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
