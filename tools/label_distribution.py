import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy import misc
import numpy as np
import math
import pickle

# labels_dict : {ind_label: count_label}
# mu : parameter to tune

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight






path_label = "/home/huangbo/HuangBo_Projects/data/nordhorn/masks/"

mypath = path_label
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

thefile = open("labeled_data.txt", 'w')
for item in onlyfiles:
    thefile.write("%s\n" % item)

file = open("labeled_data.txt")

lines = file.readlines()
nb = len(lines)
print "The number of labels are:", nb


hist_all = np.zeros(11,dtype=float)

for i in range(nb):


    image = misc.imread(path_label + lines[i][0:-1])

    for m in range(11):
        nb_label = np.count_nonzero(image == m)
        hist_all[m] = hist_all[m] +  nb_label

hist_all = hist_all/np.sum(hist_all)

print hist_all
print sum(hist_all)


plt.figure()
plt.plot(hist_all,"r*")
plt.show()



#
#
# # class_weights = {"background": hist_all[0],
# #                 "building": hist_all[1],
# #                 "asphalt / concrete": hist_all[2],
# #                 "railway": hist_all[3],
# #                 "cars": hist_all[4],
# #                 "flat vegetation": hist_all[5],
# #                 "bushes (medium vegetation)": hist_all[6],
# #                 "trees (high vegetation)": hist_all[7],
# #                 "water": hist_all[8],
# #                 "fallow land": hist_all[9],
# #                 "sand / rock": hist_all[10]}
#
# class_weights = {"0": hist_all[0],
#                 "1": hist_all[1],
#                 "2": hist_all[2],
#                 "3": hist_all[3],
#                 "4": hist_all[4],
#                 "5": hist_all[5],
#                 "6": hist_all[6],
#                 "7": hist_all[7],
#                 "8": hist_all[8],
#                 "9": hist_all[9],
#                 "10": hist_all[10]}
#
# class_weights =  create_class_weight(class_weights)
#
#
# pickle.dump(class_weights, open('class_weights', 'w'))
#
#
# class_weights = pickle.load(open('class_weights', 'r'))
# print class_weights
#
