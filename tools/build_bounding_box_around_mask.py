import os
import cv2
import yaml
import json

def convert_bbox(yml_file, annotation_folder):
    with open(yml_file, 'rb') as fp:
        spec = yaml.load(fp.read())

    masks = spec['training']['masks'] + spec['validation']['masks'] + spec['testing']['masks']
    print('The length of masks is:', len(masks))
    for _, mask in enumerate(masks[0:10]):
        mask_file = os.path.join(spec['prefix'], mask)
        mask_img = cv2.imread(mask_file)
        imgray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        annotation_data = {"img_name": mask.split('/')[-1],
                           'bboxes': []
                           }
        if len(contours) == 0:
            annotation_data['bboxes'].append(
                {'category': '',
                 'x1': '',
                 'x2': '',
                 'y1': '',
                 'y2': '',
                 })

        else:
            for i in range(len(contours)):
                cnt = contours[i]
                x, y, w, h = cv2.boundingRect(cnt)
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + w
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