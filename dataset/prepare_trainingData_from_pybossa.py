import os
import json
import pickle
import geopandas as gpd
from glob import glob


def generate_gt_from_pybossa(geojson_file, metadata, img_folder, save_folder_annotation, category_type='most_common'):

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

    for idx, _ in enumerate(img_list):
        print(idx)
        annotation_data = {"img_name": geo_data['patch_id'][idx] + '.jpg',
                           'bboxes': []
                           }

        real_coor = metadata[key_list[idx]]['geometry'].bounds
        fwd = metadata[key_list[idx]]['patch_affine']
        rev = ~fwd
        tl = rev * (real_coor[0], real_coor[1])
        br = rev * (real_coor[2], real_coor[3])
        x1 = int(tl[0])
        y1 = int(tl[1])
        x2 = int(br[0])
        y2 = int(br[1])

        annotation_data['bboxes'].append(
            {'category': int(geo_data[category_type][idx]),
             'x1': x1,
             'x2': x2,
             'y1': y1,
             'y2': y2,
             })
        with open(os.path.join(save_folder_annotation, geo_data['patch_id'][idx] + '.json'), 'w') as fp:
            json.dump(annotation_data, fp)


if __name__ == '__main__':
    geojson_file = '/home/huangbo/Building_damage/port_lavaca/output/port_lavaca.geojson'
    metadata = '/home/huangbo/Building_damage/port_lavaca/port_lavaca_patches/metadata/patch_geoinfo.pickle'
    img_folder = '/home/huangbo/Building_damage/port_lavaca/port_lavaca_patches/patches/'
    save_folder_annotation = '/home/huangbo/Building_damage/port_lavaca/port_lavaca_patches/annotations/'
    generate_gt_from_pybossa(geojson_file, metadata, img_folder, save_folder_annotation, category_type='most_common')