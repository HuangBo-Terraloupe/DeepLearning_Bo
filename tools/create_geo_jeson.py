import pickle
import numpy as np
from scipy import misc
from PIL import Image
import numpy as np

# save_label = "/home/huangbo/HuangBo_Projects/nordhorn/geo_json/"
#
# file = open("labeled_data.txt")
# lines = file.readlines()
#
# for i in range(700):
#     image = misc.imread(save_label+ lines[10][0:-4]+"tif")
#
#     print np.max(image)

import pickle
class_weights = pickle.load(open('class_weights', 'r'))
print class_weights

weights_list = [4,3,3,6,7,1,4,1,3,1,3]



for i in range(11):
    print str(i), class_weights[str(i)]

path_label = "/home/huangbo/HuangBo_Projects/data/nordhorn/labels/"
save_label = "/home/huangbo/HuangBo_Projects/nordhorn/geo_json"

file = open("labeled_data.txt")
lines = file.readlines()
nb = len(lines)


for i in range(nb):

    print "the index of image:", i
    image = misc.imread(path_label + lines[i][0:-1])
    image = np.matrix(image, dtype=float)
    print "the original image:", path_label + lines[i][0:-1]
    for j in range(11):
        np.place(image, image == j, class_weights[str(j)])

    print "the maximal value of is :", np.max(image)




    length = image.shape[0]*image.shape[1]
    nb_zero = np.count_nonzero(image)

    if length!=nb_zero:
        print "there is a zero in image"
        break

    print "the transfer image:", "/geo_json/" + lines[i][-16:-4]+"tif"
    im = Image.fromarray(image)
    im.save("geo_json/" + lines[i][0:-4]+"tif")

