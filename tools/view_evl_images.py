import os
from scipy import misc
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

image_path = "/home/huangbo/HuangBo_Projects/data/nordhorn/dataset_500/images/"
label_path = "/home/huangbo/HuangBo_Projects/data/nordhorn/dataset_500/labels/"
weights_path = "/home/huangbo/HuangBo_Projects/data/nordhorn/dataset_500/weights/"

files = [f for f in listdir(image_path) if isfile(join(image_path, f))]

for item in files:
    image = misc.imread(image_path + item)
    label = misc.imread(label_path + item[0:-3] + "png")

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)

    fig.add_subplot(1, 2, 2)
    plt.imshow(label)

    plt.show()

