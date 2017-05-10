import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import yaml


dataset_file = "/home/huangbo/objectdetection/objectdetection/huangbo_ws/nordhorn_2.yml"

with open(dataset_file) as fp:
    spec = yaml.load(fp.read())

nb = len(spec["validation"]["images"])

for i in range(nb):

    image1 = misc.imread(spec["validation"]["images"][i])
    image2 = misc.imread(spec["validation"]["labels"][i])

    print spec["validation"]["labels"][i]

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(image1)

    fig.add_subplot(1,2,2)
    plt.imshow(image2)

    plt.show()
