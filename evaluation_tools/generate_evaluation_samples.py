import os
import yaml
import numpy as np
import cv2

from keras.layers import Activation
from keras.models import Model

from models.xception_strip_1_deconv_upsample import construct
from skimage.io import imread

from models.deeplab_v3_plus import Deeplabv3
from gcloud import storage
from glob import glob

from shutil import copyfile


def preprocessing(x, mean):
    x = x.astype(np.float32)

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    x = x/255

    return np.expand_dims(x, axis=0)


number_of_validation_samples = 4000
image_extension = '.png'

yml_file = '/home/bo_huang/model_evaluation/building_validation_samples/mordor_merged_dataset_full.yml'
weights_file = '/home/bo_huang/model_evaluation/pools_validation_samples/model_class_simple_070-147.093.hdf5'
mean_file = '/home/bo_huang/model_evaluation/pools_validation_samples/mean.npy'

output_save_folder = '/home/bo_huang/model_evaluation/pools_validation_samples/predictions'
image_save_folder = '/home/bo_huang/model_evaluation/pools_validation_samples/images'
mask_save_folder = '/home/bo_huang/model_evaluation/pools_validation_samples/masks'


pool_pos_image_folder = '/home/bo_huang/model_evaluation/pools_validation_samples/12.3f3_3_class_slw_01_dev/pos_patch'
pool_pos_mask_folder = '/home/bo_huang/model_evaluation/pools_validation_samples/12.3f3_3_class_slw_01_dev/pos_label'
pool_neg_image_folder = '/home/bo_huang/model_evaluation/pools_validation_samples/12.3f3_3_class_slw_01_dev/neg_patch'
pool_neg_mask_folder = '/home/bo_huang/model_evaluation/pools_validation_samples/12.3f3_3_class_slw_01_dev/neg_label'


pos_images = glob(pool_pos_image_folder + '/*.png')
pos_masks = glob(pool_pos_mask_folder + '/*.png')
neg_images = glob(pool_neg_image_folder + '/*.png')
neg_masks = glob(pool_neg_mask_folder + '/*.png')


for i in range(2000):
    copyfile(pos_images[i], os.path.join(image_save_folder, os.path.split(pos_images[i])[-1]))
    copyfile(pos_masks[i], os.path.join(mask_save_folder, os.path.split(pos_masks[i])[-1]))
    copyfile(neg_images[i], os.path.join(image_save_folder, os.path.split(neg_images[i])[-1]))
    copyfile(neg_masks[i], os.path.join(mask_save_folder, os.path.split(neg_masks[i])[-1]))

inference_masks = glob(mask_save_folder + '/*' + image_extension)

print('filtering buildings in masks')
for id, image_path in enumerate(inference_masks):
    print(id)
    img = imread(image_path)
    img[img == 2] = 0
    cv2.imwrite(image_path, img)

# # download images from gcloud
# gclient = storage.Client()
# bucket = gclient.get_bucket('patches.terraloupe.com')


# # load images path
# with open(yml_file, 'rb') as fp:
#     spec = yaml.load(fp.read())
#
# validation_images = spec['testing']['images'][0:number_of_validation_samples]
# validation_images = [os.path.join('mordor/', x) for x in validation_images]
#
# validation_masks = spec['testing']['labels'][0:number_of_validation_samples]
# validation_masks = [os.path.join('mordor/', x) for x in validation_masks]
#
# for image_id, image in enumerate(validation_images):
#     print(image_id)
#     blob = bucket.blob(image)
#     image_name = os.path.split(image)[-1]
#     image_copy_path = os.path.join(image_save_folder, image_name)
#     blob.download_to_filename(image_copy_path)
#
# for mask_id, mask in enumerate(validation_masks):
#     print(mask_id)
#     blob = bucket.blob(mask)
#     mask_name = os.path.split(mask)[-1]
#     mask_copy_path = os.path.join(mask_save_folder, mask_name)
#     blob.download_to_filename(mask_copy_path)
#
# print('loading images and masks are done')

# id = 0
# for path in bucket.list_blobs(prefix='germany_water_masks/v0/validation/rgbi/'):
#
#     if path.name.endswith(image_extension):
#         image_name = os.path.split(path.name)[-1]
#         image_copy_path = os.path.join(image_save_folder, image_name)
#         path.download_to_filename(image_copy_path)
#         id = id + 1
#         print(id)
#
#     if id == number_of_validation_samples:
#         break
#
# id = 0
# for path in bucket.list_blobs(prefix='germany_water_masks/v0/validation/masks/'):
#
#     if path.name.endswith(image_extension):
#         mask_name = os.path.split(path.name)[-1]
#         mask_copy_path = os.path.join(mask_save_folder, mask_name)
#         path.download_to_filename(mask_copy_path)
#         id = id + 1
#         print(id)
#
#     if id == number_of_validation_samples:
#         break



# load model
# base_model = Deeplabv3(weights=None, input_tensor=None, input_shape=(400, 400, 4), classes=2, backbone='xception', OS=16)
# x = Activation(activation='softmax', name='softmax')(base_model.output)
# model = Model(input=base_model.input, output=x)

model = construct(input_shape=(400, 400), n_labels=3, n_channels=3, batch_size=None, freeze=False, weights_path=weights_file)

print(model.summary())
model.load_weights(weights_file)

# load mean
mean = np.load(mean_file)
mean = mean * 255
print('mean', mean)

inference_images = glob(image_save_folder + '/*' + image_extension)

for id, image_path in enumerate(inference_images):

    print(id)
    img = imread(image_path)
    img = preprocessing(img, mean)
    prediction = model.predict(img)

    prediction = np.argmax(prediction, axis=-1).astype('uint8')
    prediction = np.squeeze(prediction, axis= 0)

    # filter buildings
    prediction[prediction == 2] = 0

    prediction_save_path = os.path.join(output_save_folder, os.path.split(image_path)[-1].split('.')[0] + '.png')
    print(prediction_save_path)

    cv2.imwrite(prediction_save_path, prediction)