import json
import os
from glob import glob
from itertools import product

import cv2
import geopandas as gpd
import rasterio
from affine import Affine
from shapely.geometry import box
import numpy as np


def write_geojson(df, out, epsg_out):
    # write *.geojson
    header = {
        'type': 'name',
        'properties': {'name': 'urn:ogc:def:crs:EPSG::%d' % epsg_out}
    }

    result = json.loads(df.to_json())
    result['crs'] = header

    with open(out, 'w') as f:
        json.dump(result, f, indent=None, sort_keys=True)


def create_data_set(img_folder, geojson_file, data_set_name, epsg, patch_size, out_ext, save_folder_annotation,
                    save_folder_images, save_folder_negative_annotation, save_folder_negative_images,
                    geo_info_pickle_file):
    patch_num = 0
    geo_info = {}

    img_list = glob(os.path.join(img_folder, "*.tif"))
    data = gpd.read_file(geojson_file)
    data = data.to_crs(epsg=epsg)  # convert input building outlines -> correct CRS (of images!)

    print 'total img number is:', len(img_list)

    for img_idx, img in enumerate(img_list):

        # # get file name and extension separately
        # _, file_name = os.path.split(img)
        # file_name, file_ext = os.path.splitext(file_name)
        print 'current operate img:', img_idx

        # load image
        with rasterio.open(img) as fp:
            affine = fp.affine
            image = fp.read()
            # img_epsg = fp.crs

        image = image.transpose(1, 2, 0)  # flip axis to have channels last
        height, width = image.shape[0:2]

        # determine vertices within image
        n_w = range(0, height, patch_size)
        n_h = range(0, width, patch_size)
        vertices = [[y, x] for x, y in product(n_w, n_h)]  # in pixel!

        for i, v in enumerate(vertices):
            # output filename for the patch
            out_name = '%06d%s' % (patch_num, out_ext)
            out_name = data_set_name + out_name
            patch_num += 1

            x0 = v[1]
            x1 = v[1] + patch_size

            y0 = v[0]
            y1 = v[0] + patch_size

            dx = 0
            dy = 0


            if y1 > image.shape[1] or x1 > image.shape[0]:
                #print 'crop the image cross the boundry'
                continue

            if len(image.shape) > 2:
                tile = image[x0:x1, y0:y1, :]
            else:
                tile = image[x0:x1, y0:y1]

            annotation_data = {"img_name": out_name,
                               'bboxes': []
                               }

            # build geometry bounds for later analysis
            v[0] = v[0] - dy
            v[1] = v[1] - dx
            patch_affine = affine * Affine.translation(v[0], v[1])
            spoint = affine * v
            epoint = affine * (v[0] + patch_size, v[1] + patch_size)

            geo = box(spoint[0], spoint[1], epoint[0], epoint[1])

            # save for later
            geo_info[out_name] = {'patch_affine': patch_affine,
                                  'patch_size': (patch_size, patch_size)
                                  }

            # check intersection of given patch with building outlines
            intersects_flag = data.intersects(geo)

            # only proceed if intersection exists
            if sum(intersects_flag) == 0 and np.random.randint(0,11) > 5:
                annotation_data['bboxes'].append(
                    {'category': 'tank',
                     'x1': 0,
                     'x2': 1,
                     'y1': 0,
                     'y2': 1,
                     })

                file_name, file_ext = os.path.splitext(out_name)
                with open(os.path.join(save_folder_negative_annotation, file_name + '.json'), 'w') as fp:
                    json.dump(annotation_data, fp)

                # write patch
                out_path = os.path.join(save_folder_negative_images, out_name)
                cv2.imwrite(out_path, tile[:, :, ::-1])  # flip channels since opencv

            # only proceed if intersection exists
            elif sum(intersects_flag) > 0:

                # get filtered GeoDataFrame based on patch bounds
                data_filt = data.loc[intersects_flag, :]

                # iterate through all possible intersections
                for _, data_f in data_filt.iterrows():

                    # get min/max bounds of geometry (real world)
                    x_min, y_min, x_max, y_max = data_f.geometry.intersection(geo).bounds

                    # convert min/max bounds to top left/bottom right (pixel coords)
                    tl = (x_min, y_max) * ~patch_affine
                    br = (x_max, y_min) * ~patch_affine

                    # convert to top left, bottom right notation (for later analysis + training)
                    x1 = int(tl[0])
                    y1 = int(tl[1])
                    x2 = int(br[0])
                    y2 = int(br[1])

                    # filter the box that contains only a single pixel
                    if x1 == x2 or y1 == y2:
                        print 'filter a single point'
                        continue

                    # filter the box with small area
                    temp_box = box(x1, y1, x2, y2)
                    if temp_box.area < 16 * 16:
                        print 'filter a small area'
                        continue

                    category = 'tank'
                    annotation_data['bboxes'].append(
                        {'category': category,
                         'x1': x1,
                         'x2': x2,
                         'y1': y1,
                         'y2': y2,
                         })

                # only save the image patch contains a bounding box
                if len(annotation_data['bboxes']) > 0:
                    file_name, file_ext = os.path.splitext(out_name)
                    with open(os.path.join(save_folder_annotation, file_name + '.json'), 'w') as fp:
                        json.dump(annotation_data, fp)

                    # write patch
                    out_path = os.path.join(save_folder_images, out_name)
                    cv2.imwrite(out_path, tile[:, :, ::-1])  # flip channels since opencv

    with open(geo_info_pickle_file, 'wb') as fp:
        json.dump(geo_info, fp)


if __name__ == '__main__':
    # input parameters
    img_folder = '/home/huangbo/tank_detection/tank_detection/images/Regensburg'
    geojson_file = '/home/huangbo/tank_detection/tank_label/regensburg/regensburg.geojson'
    data_set_name = 'regensburg'
    patch_size = 600
    epsg = 31468  # prag: 5514, zurich: 2056, regensburg: 31468
    out_ext = '.jpg'
    save_folder_annotation = '/home/huangbo/tank_detection/dataset_negative/annotations'
    save_folder_images = '/home/huangbo/tank_detection/dataset_negative/images'
    save_folder_negative_annotation = '/home/huangbo/tank_detection/dataset_negative/annotations_negative'
    save_folder_negative_images = '/home/huangbo/tank_detection/dataset_negative/images_negative'
    geo_info_pickle_file = '/home/huangbo/tank_detection/dataset_negative/' + 'geo_info_' + data_set_name + '.pickle'

    # run
    create_data_set(img_folder, geojson_file, data_set_name, epsg, patch_size, out_ext, save_folder_annotation,
                    save_folder_images, save_folder_negative_annotation, save_folder_negative_images,
                    geo_info_pickle_file)
