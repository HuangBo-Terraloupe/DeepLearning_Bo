import os
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

gt_path = '/home/huangbo/HuangBo_Projects/regensburg/evaluation/gt/'
prediction_path = '/home/huangbo/HuangBo_Projects/regensburg/evaluation/prediction/'


mypath = "/home/huangbo/HuangBo_Projects/regensburg/evaluation/gt/"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for file in files:
    gt = gt_path + file
    prediction = prediction_path + file

    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    gt = mpimg.imread(gt)
    plt.imshow(gt)
    a.set_title('Groundtruth')

    a=fig.add_subplot(1,2,2)
    prediction = mpimg.imread(prediction)
    plt.imshow(prediction)
    a.set_title('Prediction')
    plt.show()
