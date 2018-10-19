import cv2
import click
from random import shuffle
import numpy as np
import yaml
import os

def calculate_mean(yml_file, number_samples):

    with open(yml_file, 'rb') as fp:
        spec = yaml.load(fp.read())

    image_list = spec['training']['images']
    shuffle(image_list)

    if number_samples > len(image_list):
        raise ValueError('The number of samples you choose is bigger than the whole training samples.')

    src = cv2.imread(os.path.join(spec['prefix'], image_list[0]))
    n_channels = src.shape[-1]
    image_bgr_mean = np.zeros(n_channels)
    for i, filename in enumerate(image_list[0:number_samples]):
        image = cv2.imread(os.path.join(spec['prefix'], filename))
        image = np.asarray(image, dtype=np.float32)
        image_bgr_mean += image.mean(axis=(0, 1)) / number_samples

    print(image_bgr_mean)
    return image_bgr_mean


@click.command()
@click.argument('yml_file')
@click.option('--number_samples', type=int, default = 2000)

def main(yml_file, number_samples):
    calculate_mean(yml_file, number_samples)


if __name__ == '__main__':
    main()
