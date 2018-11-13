import os
import cv2
import yaml
import json

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
                if prop['Area'] < 500:
                    continue

                # draw rectangle around segmented coins
                x1 = prop['BoundingBox'][1]
                y1 = prop['BoundingBox'][0]
                x2 = prop['BoundingBox'][4]
                y2 = prop['BoundingBox'][3]

                if x1 < 0:
                    print('negative x')
                    x1 = 0

                elif y1 < 0:
                    print('negative y')
                    y1 = 0

                elif x1 >= x2 or y1 >= y2:
                    print('wrong xy')
                    continue

                annotation_data['bboxes'].append(
                    {'category': 'parking_area',
                     'x1': x1,
                     'x2': x2,
                     'y1': y1,
                     'y2': y2,
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

    masks = masks[0:1000]

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