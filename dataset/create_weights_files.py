import cv2
import yaml
import numpy as np
from scipy import misc
from PIL import Image

weights_list = [1,8,8,12,14,1,2,1,2,1,2]

with open("/home/huangbo/HuangBo_Projects/data/nordhorn/nordhorn_total.yml") as fp:
    spec = yaml.load(fp.read())

    nb_samples = len(spec["training"]["labels"])
    print "nb_samples:", nb_samples

    path_label = "/home/huangbo/HuangBo_Projects/data/nordhorn/masks/"
    save_label = "/home/huangbo/HuangBo_Projects/data/nordhorn/weights/"
    #save_label = "/home/huangbo/Desktop/"

    for i in range(nb_samples):

        print "the index of image:", i
        image = misc.imread(spec["training"]["labels"][i])
        image = np.matrix(image, dtype=float)
        print "the original image:", spec["training"]["labels"][i]
        for j in range(11):
            np.place(image, image == j, weights_list[j])

        print "the maximal value of is :", np.max(image)

        length = image.shape[0]*image.shape[1]
        nb_zero = np.count_nonzero(image)

        if length!=nb_zero:
            print "there is a zero in image"
            break

        print "the transfer image:", save_label + spec["training"]["labels"][i][-15:-3] +"png"

        cv2.imwrite(save_label + spec["training"]["labels"][i][-15:-3] +"png", image)
        # im = Image.fromarray(image)
        # im = im.convert('RGB')
        # im.save(save_label + spec["training"]["labels"][i][-15:-3] +"png")
