import os
import cv2
import yaml
import json
import click




def data_argument(dataset_file, flip_flag, img_save_folder, json_save_folder):
    '''
    :param dataset_file:
    :param flip_flag:
    :param img_save_folder:
    :param json_save_folder:
    :return:
    '''

    with open(dataset_file, 'r') as fp:
        spec = yaml.load(fp.read())
    image_list = spec['training']['images']
    annotation_list = spec['training']['labels']

    for idx, image_file in enumerate(image_list):
        img = cv2.imread(spec['prefix'] + image_file)
        rows, cols = img.shape[:2]
        flip_img = img.copy()

        # if we do vertical flip
        if flip_flag == 'v':
            flip_img = cv2.flip(img, 0)
            _, img_name = os.path.split(image_file)
            # save the fliped image
            cv2.imwrite(os.path.join(img_save_folder, 'v_' + img_name), flip_img)

            # read the annotation and do data argumentation
            annotation = annotation_list[idx]
            with open(spec['prefix'] + annotation) as fp:
                data = json.load(fp)

            data['img_name'] = 'v_' + img_name
            for bbox in data['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

            # save the fliped annotation
            _, json_name = os.path.split(annotation)
            with open(os.path.join(json_save_folder, 'v_' + json_name), 'w') as outfile:
                json.dump(data, outfile)


        # if we do horizontal flip
        elif flip_flag == 'h':
            flip_img = cv2.flip(img, 1)
            _, img_name = os.path.split(image_file)
            # save the fliped image
            cv2.imwrite(os.path.join(img_save_folder, 'h_' + img_name), flip_img)

            # read the annotation and do data argumentation
            annotation = annotation_list[idx]
            with open(spec['prefix'] + annotation) as fp:
                data = json.load(fp)

            data['img_name'] = 'h_' + img_name
            for bbox in data['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

            # save the fliped annotation
            _, json_name = os.path.split(annotation)
            with open(os.path.join(json_save_folder, 'h_' + json_name), 'w') as outfile:
                json.dump(data, outfile)

        else:
            raise IOError('There is no argument for vertical or horizontal')

if __name__ == '__main__':
    dataset_file = '/home/huangbo/tank_detection/dataset/tank.yml'
    flip_flag = 'h'
    img_save_folder = '/home/huangbo/Desktop/data_argumentation/images'
    json_save_folder = '/home/huangbo/Desktop/data_argumentation/annotations'
    data_argument(dataset_file, flip_flag, img_save_folder, json_save_folder)