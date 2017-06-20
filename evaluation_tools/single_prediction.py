import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

from tools.load_and_transfer8GPU import load_and_transfer

from scipy import misc
from keras.models import model_from_json, load_model
from image_tools import compress_as_label,\
                        normalize_image_channelwise,\
                        predict_single,predict_complete, \
                        get_discrete_sliding_window_boxes, \
                        crop_image



class Sliding_window_evl:
    def __init__(self, image, model, box_size):
        self.image = image
        self.model = model
        self.box_size = box_size

    def prediction(self):

        boxes_list = get_discrete_sliding_window_boxes(img_size=(self.image.shape[0], self.image.shape[1]),
                                                       bbox_size=(self.box_size, self.box_size),
                                                       stride=(self.box_size / 2, self.box_size / 2))

        nb_windows = len(boxes_list)
        print "the number of windows", nb_windows
        nb_row = 0

        for i in range(nb_windows):
            if boxes_list[i][0] == 0:
                nb_row += 1
        nb_col = nb_windows / nb_row

        offset = 0
        for j in range(nb_col):

            for i in range(nb_row):
                img = crop_image(image=self.image, crop_size=(self.box_size, self.box_size),
                                 offset=(boxes_list[i + offset][0], boxes_list[i + offset][1]))

                img_norm = normalize_image_channelwise(img)
                img_norm = np.expand_dims(img_norm, axis=0)

                prediction = self.model.predict(img_norm)
                prediction = compress_as_label(prediction)
                prediction = np.reshape(prediction, (self.box_size, self.box_size))
                crop_prediction = crop_image(image=prediction, crop_size=(self.box_size / 2, self.box_size / 2),
                                             offset=(self.box_size / 4, self.box_size / 4))
                crop_prediction = np.squeeze(crop_prediction, axis=2)

                if i == 0:
                    row_image = crop_prediction
                else:
                    row_image = np.concatenate((row_image, crop_prediction), axis=1)
            offset += nb_row
            if j == 0:
                whole_image = row_image
            else:
                whole_image = np.concatenate((whole_image, row_image), axis=0)

        return whole_image

class Complete_prediction:
    def __init__(self, model, image):

        self.model = model
        self.image = image


    def prediction(self):
        normal_image = normalize_image_channelwise(self.image)
        prediction = predict_complete(self.model, normal_image)
        prediction = compress_as_label(prediction)

        return prediction



if __name__ == '__main__':
    # import matplotlib
    # matplotlib.use('Agg')

    #load the model
    json_file = '/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/06.18_resnet_256/model.json'
    weights_file = "/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/06.18_resnet_256/model.hdf5"
    model = load_and_transfer(model_file=json_file, weights_file=weights_file)

    mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/images_subset/"
    image_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    #for image in image_files:
    evl_image = mypath + image_files[4]
    evl_image = misc.imread(evl_image)

    #sl_evl = Sliding_window_evl(evl_image, model, 256)
    sl_evl = Complete_prediction(model, evl_image)
    prediction = sl_evl.prediction()
    #
    # image_index = path + image[0:-3] + "tif"

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(evl_image)

    fig.add_subplot(1, 2, 2)
    plt.imshow(prediction)

    plt.show()
