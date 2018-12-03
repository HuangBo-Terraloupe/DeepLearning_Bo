import os
import json
import multiprocessing

from glob import glob

def generate_bbox(image_list, annotation_folder):

    for _, mask in enumerate(image_list):

        annotation_data = {"img_name": mask.split('/')[-1],
                           'bboxes': []
                           }

        annotation_data['bboxes'].append(
            {'category': '',
             'x1': '',
             'x2': '',
             'y1': '',
             'y2': '',
             })

        # save annotation
        with open(os.path.join(annotation_folder, mask.split('/')[-1].split('.')[0] + '.json'), 'w') as fp:
            json.dump(annotation_data, fp)


def convert_bbox(image_folder, annotation_folder, n_worker):
    '''
    Args:
        image_folder:
        annotation_folder:
        n_worker:

    Returns:

    '''
    background_images = glob(image_folder + '/' + 'background*' + '.tif')
    highway_images = glob(image_folder + '/' + 'highway*' + '.tif')

    images = background_images + highway_images
    jobs = []

    total_num_images = len(images)
    print('total number of images:', total_num_images)
    image_for_each_worker = int(total_num_images / n_worker)

    for i in range(n_worker):
        if i != n_worker -1:
            p = multiprocessing.Process(
                target=generate_bbox,
                args=(images[i*image_for_each_worker:(i+1)*image_for_each_worker], annotation_folder,))
        else:
            p = multiprocessing.Process(
                target=generate_bbox,
                args=(images[i*image_for_each_worker:], annotation_folder,))
        jobs.append(p)
        p.start()



if __name__ == '__main__':
    image_folder = '/mnt/disks/conti-parking/germany_v2/images'
    annotation_folder = '/mnt/disks/conti-parking/germany_v2/annotations'
    n_worker = 16
    convert_bbox(image_folder, annotation_folder, n_worker)