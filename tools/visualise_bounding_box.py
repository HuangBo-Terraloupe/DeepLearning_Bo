import os
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from glob import glob

import matplotlib.pyplot as plt

def visualise_bboxes(annotation_folder, image_folder):
    annotations = glob(annotation_folder + '*.json')
    for annotation in annotations:
        with open(annotation, 'r') as f:
            data = json.load(f)
            img_name = data['img_name']
            img_path = os.path.join(image_folder, img_name)
            img = np.array(Image.open(img_path), dtype=np.uint8)
            # Create figure and axes
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            for bbox in data['bboxes']:
                rect = patches.Rectangle((bbox['x1'], bbox['y1']), bbox['x2']-bbox['x1'], bbox['y2']-bbox['y1'],
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()



if __name__ == '__main__':
    annotation_folder = '/home/huangbo/Building_damage_nofilter/annotations/'
    image_folder = '/home/huangbo/Building_damage_nofilter/images/'
    visualise_bboxes(annotation_folder, image_folder)
