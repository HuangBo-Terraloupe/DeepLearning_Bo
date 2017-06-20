import os
from os import listdir
from os.path import isfile, join
from scipy import misc

mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/dataset_500/weights/"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for item in files:
    image = misc.imread(mypath + item)
    if image.shape != (500, 500):
        os.remove(mypath + item)



