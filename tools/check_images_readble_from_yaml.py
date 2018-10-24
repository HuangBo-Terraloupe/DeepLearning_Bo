import os
import yaml
import cv2
import multiprocessing


def check_images(prefix, images_list, worker_id):
    wrong_images = []
    for i, file in enumerate(images_list):
        image = cv2.imread(os.path.join(prefix, file))
        if image is None:
            print('found empty image:', os.path.join(prefix, file))
            wrong_images.append(os.path.join(prefix, file))
        elif len(image.shape) == 3:
            continue
        else:
            print('found wrong channel image:', os.path.join(prefix, file))
            wrong_images.append(os.path.join(prefix, file))
        if i % 500 ==0:
            print(i)
    with open('/home/bo_huang/wrongimages' + str(worker_id) + '.txt', 'w') as f:
        for item in wrong_images:
            f.write("%s\n" % item)

def run(yml_file, n_worker):
    jobs = []
    with open(yml_file, 'rb') as fp:
        spec = yaml.load(fp.read())

    image_list = spec['training']['images'] + spec['validation']['images'] + spec['testing']['images']
    total_num_images = len(image_list)
    image_for_each_worker = int(total_num_images / n_worker)
    for i in range(n_worker):
        if i != n_worker -1:
            p = multiprocessing.Process(
                target=check_images,
                args=(spec['prefix'],image_list[i*image_for_each_worker:(i+1)*image_for_each_worker], i,))
        else:
            p = multiprocessing.Process(
                target=check_images,
                args=(spec['prefix'],image_list[i*image_for_each_worker:], i))
        jobs.append(p)
        p.start()


if __name__ == '__main__':
    yml_file = '/data/here/houston.yml'
    n_worker = 12
    run(yml_file, n_worker)
