from os import listdir
from os.path import isfile, join

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from keras.models import model_from_json, load_model
from image_tools import compress_as_label,normalize_image_channelwise,predict_single,predict_complete, crop_image
from sliding_window_evaluation import Sliding_window_evl

class Show_evl:

    def __init__(self, model, image):
        self.model = model
        self.image = image

    def evl_complete(self):

        prediction = predict_complete(self.model, self.image)
        #print "the prediction shape after complete predict", prediction.shape
        prediction = compress_as_label(prediction)
        #print "the final prediction shape", prediction.shape

        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(self.image)

        fig.add_subplot(1,2,2)
        plt.imshow(prediction)

        plt.show()

    def evl_sliding(self, box_size):

        sl_evl = Sliding_window_evl(self.model, box_size)
        prediction = sl_evl.evluation_single(self.image)
        crop_shape = prediction.shape
        self.image = crop_image(image=self.image, crop_size=crop_shape, offset=(box_size / 4, box_size / 4))

        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(self.image)

        fig.add_subplot(1,2,2)
        plt.imshow(prediction)

        plt.show()


if __name__ == "__main__":

    # model
    json_file = open('/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/model_540.json', 'r')
    # json_file = open("model.json",'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/05.15_unet_540/weights_best.hdf5")


    # # images
    # dataset_file = "/home/huangbo/objectdetection/objectdetection/huangbo_ws/nordhorn_2.yml"
    # with open(dataset_file) as fp:
    #     spec = yaml.load(fp.read())
    #
    # nb_samples = len(spec["validation"]["images"])
    #
    # index = np.random.randint(0, nb_samples)
    # print "the index is", index
    #
    # evl_image = spec["validation"]["images"][index]
    # evl_image = misc.imread(evl_image)


    mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/building_images/"
    image_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for image in image_files:

        evl_image = mypath + image
        evl_image = misc.imread(evl_image)

        show_image = Show_evl(model,evl_image)
        #show_image.evl_sliding(box_size=540)
        show_image.evl_complete()

    # nb_images = len(image_files)
    #
    # for i in range(nb_images):
    #
    #     evl_image = mypath + image_files[i+10]
    #     evl_image = misc.imread(evl_image)
    #
    #     show_image = Show_evl(model,evl_image)
    #     show_image.evl_sliding(box_size=540)
    #     #show_image.evl_complete()
