import cv2
import multiprocessing

from glob import glob

def downsample_images(image_list, downsample_size):
    for i, file in enumerate(image_list):
        print(i)
        item = cv2.imread(file)
        cv2.imwrite(file, cv2.resize(item, downsample_size))


def run(list_image_folder, downsample_size, n_worker):
    jobs = []
    files = []

    for folder in list_image_folder:
        files = files + glob(folder + '*.png')

    image_for_each_worker = int(len(files) / n_worker)
    for i in range(n_worker):
        if i != n_worker -1:
            p = multiprocessing.Process(
                target=downsample_images,
                args=(files[i*image_for_each_worker:(i+1)*image_for_each_worker], downsample_size,))
        else:
            p = multiprocessing.Process(
                target=downsample_images,
                args=(files[i*image_for_each_worker:], downsample_size, ))
        jobs.append(p)
        p.start()


if __name__ == '__main__':
    n_worker = 4
    image_folder = '/home/terraloupe/Dataset/30cm_data/houston/images/'
    mask_folder = '/home/terraloupe/Dataset/30cm_data/houston/masks/'
    folder_list = [image_folder, mask_folder]
    downsample_size = (300, 300)
    run(folder_list, downsample_size, n_worker)
