import os
from glob import glob
from shutil import copyfile


img_folder = '/home/bo_huang/lane_marking_dataset/nrw_old/images/'
mask_folder = '/home/bo_huang/lane_marking_dataset/nrw_old/lane_marking/masks/'

dst_img = '/home/bo_huang/lane_marking_dataset/combined/images'
dst_mask = '/home/bo_huang/lane_marking_dataset/combined/masks'

src_images = glob(img_folder + '*.jpg')
src_masks = glob(mask_folder + '*.tif')

for img in src_images:
    file_name = os.path.split(img)[-1]
    dst = os.path.join(dst_img, file_name)
    copyfile(img, dst)

for mask in src_masks:
    file_name = os.path.split(mask)[-1]
    dst = os.path.join(dst_mask, file_name)
    copyfile(mask, dst)






img_folder = '/home/bo_huang/lane_marking_baseline/images/lanes/'
mask_folder = '/home/bo_huang/lane_marking_baseline/masks/lanes/'

img_extension = '.tif'
mask_extension = '.tif'


images = glob(img_folder + '*' + img_extension)
masks = glob(mask_folder + '*' + mask_extension)

print('number of images:', len(images), 'number of masks:', len(masks))

images = [os.path.split(x)[-1] for x in images]
masks = [os.path.split(x)[-1] for x in masks]


common_files = list(set(images).intersection(masks))


for mask in masks:
    if mask not in common_files:
        print(os.path.join(mask_folder, mask))
        os.remove(os.path.join(mask_folder, mask))