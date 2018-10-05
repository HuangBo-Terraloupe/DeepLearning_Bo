import cv2
from glob import glob
import click
from random import shuffle
import numpy as np

def calculate_mean(image_folder, image_extension, number_samples):
    images = glob(image_folder + '/*.' + str(image_extension))
    shuffle(images)

    if number_samples > len(images):
        raise ValueError('The number of samples you choose is bigger than the whole training samples.')

    src = cv2.imread(images[0])
    n_channels = src.shape[-1]
    image_bgr_mean = np.zeros(n_channels)
    for i, filename in enumerate(images[0:number_samples]):
        image = cv2.imread(filename)
        image = np.asarray(image, dtype=np.float32)
        image_bgr_mean += image.mean(axis=(0, 1)) / number_samples

    print(image_bgr_mean)
    return image_bgr_mean


@click.command()
@click.argument('image_folder')
@click.argument('image_extension')
@click.option('--number_samples', type=int, default = 2000)

def main(image_folder, image_extension, number_samples):
    calculate_mean(image_folder, image_extension, number_samples)


if __name__ == '__main__':
    main()
