import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from keras.models import model_from_json, load_model
from image_tools import compress_as_label,normalize_image_channelwise,predict_single,predict_complete
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

        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(self.image)

        fig.add_subplot(1,2,2)
        plt.imshow(prediction)

        plt.show()


if __name__ == "__main__":

    # model
    # json_file = open('/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/model_540.json', 'r')
    # # json_file = open("model.json",'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights("/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/05.10/weights_best.hdf5")

    from models import fcn8_vgg
    model = fcn8_vgg.Fcn_8(batch_size=1, input_shape=(240,240), n_channels=3, no_classes=11)
    model = model.build_model()
    model.load_weights("/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/fcn_05.12_240/weights_end.hdf5")

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

    mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/images_evl"

    f = open("/home/huangbo/objectdetection/objectdetection/huangbo_ws/evaluation.txt")
    lines = f.readlines()

    index = np.random.randint(0,len(lines))
    print "the index is", index

    #index = 1501
    evl_image = mypath + "/" + lines[index][0:15]
    evl_image = misc.imread(evl_image)


    show_image = Show_evl(model,evl_image)
    #show_image.evl_sliding(box_size=540)
    show_image.evl_complete()
