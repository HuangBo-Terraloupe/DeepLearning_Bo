import os
import yaml
import cv2
import multiprocessing

from glob import glob


def check_images(images_list):
    wrong_images = []
    for i, file in enumerate(images_list):
        image = cv2.imread(file)
        if image is None:
            print('found empty image:', file)
            wrong_images.append(file)
        elif len(image.shape) == 3:
            continue
        else:
            print('found wrong channel image:', file)
            wrong_images.append( file)
        if i % 1000 ==0:
            print(i)
    with open('/home/bo_huang/wrongimages.txt', 'w') as f:
        for item in wrong_images:
            f.write("%s\n" % item)

def run(input_file, n_worker):
    jobs = []
    image_list = []

    if type(input_file) is list:
        print('get images from folder ...')
        for folder in input_file:
            image_list = image_list + glob(folder + '.png')

    else:
        print('get images from yaml ...')
        with open(input_file, 'rb') as fp:
            spec = yaml.load(fp.read())
        image_list = spec['training']['images'] + spec['validation']['images'] + spec['testing']['images']
        image_list = [os.path.join(spec['prefix'], f) for f in image_list]

    total_num_images = len(image_list)
    print('total number of images and masks:', total_num_images)
    image_for_each_worker = int(total_num_images / n_worker)
    for i in range(n_worker):
        if i != n_worker -1:
            p = multiprocessing.Process(
                target=check_images,
                args=(image_list[i*image_for_each_worker:(i+1)*image_for_each_worker],))
        else:
            p = multiprocessing.Process(
                target=check_images,
                args=(image_list[i*image_for_each_worker:],))
        jobs.append(p)
        p.start()


if __name__ == '__main__':
    image_folder_1 = '/data/here/houston/images_30/'
    image_folder_2 = '/data/here/san_francisco/images_30/'
    input_file = [image_folder_1, image_folder_2]

    n_worker = 64
    run(input_file, n_worker)
