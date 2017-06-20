import numpy as np

from os import listdir
from os.path import isfile, join

from scipy import misc
from keras.models import model_from_json, load_model
from image_tools import compress_as_label,normalize_image_channelwise,predict_single,predict_complete, \
    get_discrete_sliding_window_boxes, crop_image


import matplotlib.pyplot as plt


class Sliding_window_evl:
    def __init__(self, image, model, box_size):
        self.image = image
        self.model = model
        self.box_size = box_size

    def evluation(self):

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
#--------------------------------------------------------------------------------------------------------------------------------#

json_file = open('/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/model_540.json', 'r')
#json_file = open("model.json",'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/05.13_unet_540/weights_best.hdf5")

# # ----------------------------------------------evaluate and save the image------------------------------------------------------#
# path = "/home/huangbo/Desktop/evl_05.13_noSliding/"
# #os.mkdir(path)
#
# mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/building_images/"
# image_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
# for image in image_files:
#
#
#     evl_image = mypath + image
#     evl_image = misc.imread(evl_image)
#
#     #image_down = transform.resize(evl_image, (evl_image.shape[0] // 2, evl_image.shape[1] // 2))
#     evl_image_ = normalize_image_channelwise(evl_image)
#
#
#     prediction = predict_complete(model, evl_image_)
#     prediction = compress_as_label(prediction)
#
#
#
#     image_index = path + image[0:-3] + "tif"
#
#     fig = plt.figure()
#     fig.add_subplot(1,2,1)
#     plt.imshow(evl_image)
#
#     fig.add_subplot(1,2,2)
#     plt.imshow(prediction)
#
#     fig.savefig(image_index)
#     plt.close("all")

# ----------------------------------------------evaluate and show the image-------------------------------------------------------#
# index = np.random.randint(0, len(lines))
# print "the index is", index
#
#
# evl_image = mypath + "/" + lines[index][0:15]
# evl_image = misc.imread(evl_image)
#
# # downsample-image
# #image_down = transform.resize(evl_image, (evl_image.shape[0] // 2, evl_image.shape[1] // 2))
#
# prediction = predict_complete(model, evl_image)
# print "the prediction shape after complete predict", prediction.shape
# prediction = compress_as_label(prediction)
# print "the final prediction shape", prediction.shape
#
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(evl_image)
#
# fig.add_subplot(1,2,2)
# plt.imshow(prediction)
#
# plt.show()
# ------------------------------evaluation based on sliding window----------------------------------------------#

mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/building_images/"
image_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for image in image_files:

    evl_image = mypath + image
    evl_image = misc.imread(evl_image)

    sl_evl = Sliding_window_evl(evl_image, model, 256)
    prediction = sl_evl.evluation()
    #
    # image_index = path + image[0:-3] + "tif"

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(evl_image)

    fig.add_subplot(1,2,2)
    plt.imshow(prediction)

    plt.show()

    # fig.savefig(image_index)
    # plt.close("all")

# -----------------------------evaluation based on sliding window----------------------------#


