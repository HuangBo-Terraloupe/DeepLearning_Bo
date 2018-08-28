import os
import cv2
import json
import pickle
import geopandas as gpd
from glob import glob
import click


def generate_gt_from_pybossa(geojson_file, metadata, img_folder, save_folder_annotation, category_type='most_common', img_save_folder=None):

    # load geojson data
    geo_data = gpd.read_file(geojson_file)

    # load metadata
    with open(metadata, 'rb') as fp:
        metadata = pickle.load(fp)
    key_list = list(metadata.keys())
    key_list.sort()

    # load all the images
    img_list = glob(img_folder + '*.jpg')
    img_list.sort()

    for idx, image_name in enumerate(img_list[0:100]):
        print(idx)

        real_coor = metadata[key_list[idx]]['geometry'].bounds
        fwd = metadata[key_list[idx]]['patch_affine']
        rev = ~fwd
        tl = rev * (real_coor[0], real_coor[3])
        br = rev * (real_coor[2], real_coor[1])

        x1 = int(tl[0])
        y1 = int(tl[1])
        x2 = int(br[0])
        y2 = int(br[1])

        # check if the coordinates are correct
        image = cv2.imread(image_name)
        img_width = image.shape[1]
        img_height = image.shape[0]

        if y1 > img_height or y2 > img_height or x1 > img_width or x2 > img_width:
            print('the bounding box is cross the boundary of image! skip this image.')
            continue
        if x1 >= x2 or y1 >= y2:
            print('the coordinates are fliped! skip this image.')
            continue

        if save_folder_annotation is not None:
            annotation_data = {"img_name": geo_data['patch_id'][idx] + '.jpg',
                               'bboxes': []
                               }

            annotation_data['bboxes'].append(
                {'category': int(geo_data[category_type][idx]),
                 'x1': x1,
                 'x2': x2,
                 'y1': y1,
                 'y2': y2,
                 })
            with open(os.path.join(save_folder_annotation, geo_data['patch_id'][idx] + '.json'), 'w') as fp:
                json.dump(annotation_data, fp)

        if img_save_folder is not None:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            img_save_path = os.path.join(img_save_folder, geo_data['patch_id'][idx] + '.jpg')
            cv2.imwrite(img_save_path, image)


if __name__ == '__main__':
    geojson_file = '/home/terraloupe/Building_damage/Keys/output/keys.geojson'
    metadata = '/home/terraloupe/Building_damage/Keys/metadata/patch_geoinfo.pickle'
    img_folder = '/home/terraloupe/Building_damage/Keys/patches/'
    save_folder_annotation = None #'/home/huangbo/Building_damage/port_lavaca/port_lavaca_patches/annotations/'
    category_type = 'most_common'
    img_save_folder = '/home/terraloupe/Desktop/Building_train_samples/keys'
    generate_gt_from_pybossa(geojson_file, metadata, img_folder, save_folder_annotation, category_type, img_save_folder)