import os
import cv2
import json
import rasterio
import geopandas as gpd

from glob import glob
from affine import Affine
from itertools import product
from shapely.geometry import box, Polygon


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


def create_data_set(img_folder, geojson_file, patch_size, out_ext, save_folder_annotation, save_folder_images,
                    geo_info_pickle_file):
    patch_num = 0
    geo_info = {}

    img_list = glob(img_folder + "*.tif")
    data = gpd.read_file(geojson_file)
    data = data.to_crs(epsg=32614)  # convert input building outlines -> correct CRS (of images!)

    for img in img_list:

        # # get file name and extension separately
        # _, file_name = os.path.split(img)
        # file_name, file_ext = os.path.splitext(file_name)

        # load image
        with rasterio.open(img) as fp:
            affine = fp.affine
            image = fp.read()
            img_epsg = fp.crs

        image = image.transpose(1, 2, 0)  # flip axis to have channels last
        height, width = image.shape[0:2]

        # determine vertices within image
        n_w = range(0, width, patch_size)
        n_h = range(0, height, patch_size)
        vertices = [[y, x] for x, y in product(n_w, n_h)]  # in pixel!

        for i, v in enumerate(vertices):
            if len(image.shape) > 2:
                tile = image[v[1]:v[1] + patch_size, v[0]:v[0] + patch_size, :]
            else:
                tile = image[v[1]:v[1] + patch_size, v[0]:v[0] + patch_size]

            # output filename for the patch
            out_name = '%06d%s' % (patch_num, out_ext)
            patch_num += 1
            print 'idx', patch_num

            annotation_data = {"img_name": out_name,
                               'bboxes': []
                               }

            # build geometry bounds for later analysis
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
            if sum(intersects_flag) > 0:

                # get filtered GeoDataFrame based on patch bounds
                data_filt = data.loc[intersects_flag, :]

                # iterate through all possible intersections
                for _, data_f in data_filt.iterrows():

                    x1, y1, x2, y2 = data_f.geometry.intersection(geo).bounds  # bounds in real world coords

                    tl = (x1, y2) * ~patch_affine
                    br = (x2, y1) * ~patch_affine
                    x1 = int(tl[0])
                    y1 = int(tl[1])
                    x2 = int(br[0])
                    y2 = int(br[1])

                    temp_poly = Polygon([[x1, y1],
                                         [x1, y2],
                                         [x2, y2],
                                         [x2, y1]])

                    area = temp_poly.area
                    print 'area:', area
                    if area < 20*20:
                        print 'filter a very small roof area'
                        continue

                    if x1 == x2 and y1 == y2:
                        print "x1 == x2, y1==y2, single point"
                        continue

                    try:
                        ratio = float(x2 - x1) / float(y2 - y1)
                    except:
                        ratio = float(y2 - y1) / float(x2 - x1)

                    print 'ratio:', ratio

                    if ratio > 40 or ratio < 1.0/40.0:
                        print 'filter a wrong ratio roof area'
                        continue

                    category = data['damage_class'][int(intersects_flag[intersects_flag==True].index[0])]
                    annotation_data['bboxes'].append(
                        {'category': category,
                         'x1': int(tl[0]),
                         'x2': int(br[0]),
                         'y1': int(tl[1]),
                         'y2': int(br[1]),
                         })

                file_name, file_ext = os.path.splitext(out_name)
                with open(save_folder_annotation + file_name + '.json', 'w') as fp:
                    json.dump(annotation_data, fp)

                # write patch
                out_path = os.path.join(save_folder_images, out_name)
                cv2.imwrite(out_path, tile[:, :, ::-1])  # flip channels since opencv

    with open(geo_info_pickle_file, 'w') as fp:
        json.dump(geo_info, fp)


if __name__ == '__main__':

    # input parameters
    img_folder = '/home/huangbo/HuangBo_Projects/harvey/focus_aoi/'
    geojson_file = '/home/huangbo/HuangBo_Projects/harvey/AZHAR.geojson'
    patch_size = 500
    out_ext = '.jpg'
    save_folder_annotation = '/home/huangbo/HuangBo_Projects/harvey/training_data_filter_object/annotations/'
    save_folder_images = '/home/huangbo/HuangBo_Projects/harvey/training_data_filter_object/images/'
    geo_info_pickle_file = '/home/huangbo/HuangBo_Projects/harvey/training_data_filter_object/geo_info.json'

    # run
    create_data_set(img_folder, geojson_file, patch_size, out_ext, save_folder_annotation, save_folder_images,
                    geo_info_pickle_file)