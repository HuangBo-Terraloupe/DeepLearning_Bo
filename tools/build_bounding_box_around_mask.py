import os
import cv2
import yaml
import json
import numpy as np
import multiprocessing

from skimage.measure import label, regionprops

def generate_bbox(mask_list, prefix, annotation_folder):

    for _, mask in enumerate(mask_list):
        mask_file = os.path.join(prefix, mask)
        mask_img = cv2.imread(mask_file)

        annotation_data = {"img_name": mask.split('/')[-1],
                           'bboxes': []
                           }

        if mask_img is None:
            raise ValueError('The mask file does not exist!')

        if mask_img.any() == 0:
            annotation_data['bboxes'].append(
                {'category': '',
                 'x1': '',
                 'x2': '',
                 'y1': '',
                 'y2': '',
                 })

        else:
            lbl_0 = label(mask_img)
            props = regionprops(lbl_0)

            for prop in props:
                # skip small images
                if prop['Area'] < 200:
                    continue

                # draw rectangle around segmented coins
                minr = prop['BoundingBox'][0]
                minc = prop['BoundingBox'][1]
                maxr = prop['BoundingBox'][3]
                maxc = prop['BoundingBox'][4]

                if minr < 0:
                    print('negative x')
                    minr = 0

                elif minc < 0:
                    print('negative y')
                    minr = 0

                elif minr >= maxr or minc >= maxc:
                    print('wrong xy')
                    continue

                annotation_data['bboxes'].append(
                    {'category': 'parking_area',
                     'x1': minr,
                     'x2': minc,
                     'y1': maxr,
                     'y2': maxc,
                     })

        # save annotation
        with open(os.path.join(annotation_folder, mask.split('/')[-1].split('.')[0] + '.json'), 'w') as fp:
            json.dump(annotation_data, fp)



def convert_bbox(yml_file, annotation_folder, n_worker):
    with open(yml_file, 'rb') as fp:
        spec = yaml.load(fp.read())

    masks = spec['training']['labels'] + spec['validation']['labels'] + spec['testing']['labels']
    prefix = spec['prefix']
    jobs = []

    total_num_images = len(masks)
    print('total number of images and masks:', total_num_images)
    image_for_each_worker = int(total_num_images / n_worker)

    for i in range(n_worker):
        if i != n_worker -1:
            p = multiprocessing.Process(
                target=generate_bbox,
                args=(masks[i*image_for_each_worker:(i+1)*image_for_each_worker], prefix, annotation_folder,))
        else:
            p = multiprocessing.Process(
                target=generate_bbox,
                args=(masks[i*image_for_each_worker:], prefix, annotation_folder,))
        jobs.append(p)
        p.start()



if __name__ == '__main__':
    yml_file = '/home/bo_huang/parking_bbox_detection.yml'
    annotation_folder = '/mnt/disks/conti-parking/germany_v2/annotations'
    n_worker = 24
    convert_bbox(yml_file, annotation_folder, n_worker)