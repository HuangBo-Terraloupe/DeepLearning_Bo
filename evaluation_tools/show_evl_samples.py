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
    json_file = open('/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/model_252.json', 'r')
    # json_file = open("model.json",'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/05.09/weights_best.hdf5")

    # images
    dataset_file = "/home/huangbo/objectdetection/objectdetection/huangbo_ws/nordhorn_2.yml"
    with open(dataset_file) as fp:
        spec = yaml.load(fp.read())

    nb_samples = len(spec["validation"]["images"])

    index = np.random.randint(0, nb_samples)
    print "the index is", index

    evl_image = spec["validation"]["images"][index]
    evl_image = misc.imread(evl_image)

    show_image = Show_evl(model,evl_image)
    show_image.evl_sliding(box_size=252)
