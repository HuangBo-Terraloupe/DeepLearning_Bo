import cv2
import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

from tools.load_and_transfer8GPU import load_and_transfer

from scipy import misc
from keras.models import model_from_json, load_model
from image_tools import compress_as_label,\
                        normalize_image_channelwise,\
                        predict_single,\
                        predict_complete, \
                        get_discrete_sliding_window_boxes, \
                        crop_image, \
                        discrete_matshow



class Sliding_window_evl:
    def __init__(self, image, model, box_size, mean):
        self.image = image
        self.model = model
        self.box_size = box_size
        self.mean = mean

    def prediction(self):

        boxes_list = get_discrete_sliding_window_boxes(img_size=(self.image.shape[0],
                                                                 self.image.shape[1]),
                                                       bbox_size=(self.box_size, self.box_size),
                                                       stride=(self.box_size / 2, self.box_size / 2)
                                                       )

        nb_windows = len(boxes_list)
        nb_row = 0

        for i in range(nb_windows):
            if boxes_list[i][0] == 0:
                nb_row += 1
        nb_col = nb_windows / nb_row

        offset = 0
        for j in range(nb_col):

            for i in range(nb_row):
                img = crop_image(image= self.image,
                                 crop_size=(self.box_size, self.box_size),
                                 offset=(boxes_list[i + offset][0], boxes_list[i + offset][1])
                                 )

                # img_norm = img - self.mean
                # img_norm = img_norm / 255.

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
    def __init__(self, model, image, mean=None):

        self.model = model
        self.image = image
        self.mean = mean

    def prediction(self):

        if self.mean == None:
            normal_image = normalize_image_channelwise(self.image)

        else:
            normal_image = self.image - self.mean
            normal_image = normal_image / 255.

        prediction = predict_complete(self.model, normal_image)
        prediction = compress_as_label(prediction)

        return prediction



if __name__ == '__main__':

    # #load the model
    # json_file = '/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/06.18_resnet_256/model.json'
    # weights_file = "/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/06.18_resnet_256/model.hdf5"
    # model = load_and_transfer(model_file=json_file, weights_file=weights_file)

    # model
    json_file = open('/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/model_540.json', 'r')
    # json_file = open("model.json",'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/05.11_unet_540/weights_best.hdf5")

    # mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/images_evl/"
    # image_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #
    # #for image in image_files:
    # evl_image = mypath + image_files[np.random.randint(1, 100)]

    image_path = '/home/huangbo/Downloads/tom_006.jpg'
    evl_image = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

    sl_evl = Sliding_window_evl(evl_image, model, 540, [114.96066323, 116.50405346, 102.74354111])
    #sl_evl = Complete_prediction(model, evl_image)
    prediction = sl_evl.prediction()

    # dic = {'background': 0,
    #        'building': 1,
    #        'concrete': 2,
    #        'railway': 3,
    #        'cars': 4,
    #        'flat vegetation': 5,
    #        'bushes (medium vegetation)': 6,
    #        'trees (high vegetation)': 7,
    #        'water': 8,
    #        'fallow land': 9,
    #        'sand / rock': 10
    #        }

    # dic = {0:'background',
    #        1:'building',
    #        2:'concrete',
    #        3:'railway',
    #        4:'cars',
    #        5:'flat vegetation',
    #        6:'bushes (medium vegetation)',
    #        7:'trees (high vegetation)',
    #        8:'water',
    #        9:'fallow land',
    #        10:'sand / rock'
    #        }

    dic = ['background',
           'building',
           'concrete',
           'railway',
           'cars',
           'flat vegetation',
           'bushes',
           'trees',
           'water',
           'fallow land',
           'sand / rock']
    discrete_matshow(prediction, 11, dic)

    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(evl_image)
    #
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(prediction)
    #
    # plt.show()
