import numpy as np
import yaml
import os

from scipy import misc
#from skimage import transform
from keras.models import model_from_json, load_model
from image_tools import compress_as_label,normalize_image_channelwise,predict_single,predict_complete
from sliding_window_evaluation import Sliding_window_evl

import matplotlib.pyplot as plt
#--------------------------------------------------------------------------------------------------------------------------------#

# mypath = "/home/huangbo/HuangBo_Projects/data/nordhorn/images_evl"

# f = open("evaluation.txt")
# lines = f.readlines()

#--------------------------------------------------------------------------------------------------------------------------------#

json_file = open('/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/05.03/model.json', 'r')
#json_file = open("model.json",'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/05.06/weights_best.hdf5")

# # ----------------------------------------------evaluate and save the image------------------------------------------------------#
#path = "/home/huangbo/Desktop/evl_05.03_noSliding/"
#os.mkdir(path)

# for i in range(100):

#     index = np.random.randint(0,len(lines))
#     print "the index is", index

#     evl_image = mypath + "/" + lines[index][0:15]
#     evl_image = misc.imread(evl_image)

#     #image_down = transform.resize(evl_image, (evl_image.shape[0] // 2, evl_image.shape[1] // 2))
#     evl_image_ = normalize_image_channelwise(evl_image)


#     prediction = predict_complete(model, evl_image_)
#     prediction = compress_as_label(prediction)



#     image_index = "/home/huangbo/Desktop/evl_05.03_noSliding/" + str(index) + ".png"

#     fig = plt.figure()
#     fig.add_subplot(1,2,1)
#     plt.imshow(evl_image)

#     fig.add_subplot(1,2,2)
#     plt.imshow(prediction)

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


# index = np.random.randint(0, len(lines))
# print "the index is", index

# evl_image = mypath + "/" + lines[index][0:15]
# evl_image = misc.imread(evl_image)

# sl_evl = Sliding_window_evl(evl_image, model, 32*10+220)
# prediction = sl_evl.evluation()

# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(evl_image)

# fig.add_subplot(1,2,2)
# plt.imshow(prediction)

# plt.show()

# -----------------------------evaluation based on sliding window----------------------------#


