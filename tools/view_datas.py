import matplotlib.pyplot as plt
from scipy import misc
import numpy as np


# from os import listdir
# from os.path import isfile, join

# mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/labels"
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# thefile = open("labeled_data.txt", 'w')
# for item in onlyfiles:
#     thefile.write("%s\n" % item)


path_train = "/home/huangbo/HuangBo_Projects/data/nordhorn/images/"

path_label = "/home/huangbo/HuangBo_Projects/data/nordhorn/labels/"

file = open("labeled_data.txt")

hist_all = np.zeros(11,dtype=None)

while 1:

	line = file.readline()
	view_image = line[0:11]
	print line

	image1 = misc.imread(path_train + view_image + ".jpg")
	#print image1.shape
	image2 = misc.imread(path_label + view_image + ".png")
	#print image2.shape

	#for m in range(11):

	#	nb_label = np.count_nonzero(image2 == m)
	#	hist_all[m] = hist_all[m] + nb_label

	#print hist_all


	fig = plt.figure()
	fig.add_subplot(1,2,1)
	plt.imshow(image1)

	fig.add_subplot(1,2,2)
	plt.imshow(image2)

	plt.show()


	print image1.shape
	print type(image1)
	print image1.dtype
