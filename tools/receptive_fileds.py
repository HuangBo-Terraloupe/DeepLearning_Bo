def nb_out_features(nb_features, p, k, s):
    return int((nb_features + 2*p -k)/s)+1

def distance_2pixel(jump_in, s):
    return jump_in*s

def receptive_field_size(r_in, k, jump):
    return r_in + (k-1)*jump

def start_point(start_in, p, k, jump):
    return start_in + ((k-1)/2 - p)*jump

'''

input : 
nb_features = image_size
r=1
n=1
start = 0.5

'''
# image_size = 9
#
# layer1 = nb_out_features(nb_features=image_size, p=1, k=3, s=2)
# jump1 = distance_2pixel(jump_in=1, s=2)
# rf_1 = receptive_field_size(r_in=1, k=3, jump=1)
# start_1 = start_point(start_in=0.5, p=1, k=3, jump=1)
#
# print "number of features:", layer1
# print "jump:", jump1
# print "the receptive filed size is:", rf_1
# print "the starting point:", start_1
#
# print "------------------------------------"
#
# layer2 = nb_out_features(layer1, 1, 3, 2)
# jump2 = distance_2pixel(jump1, 2)
# rf_2 = receptive_field_size(rf_1, 3, jump1)
# start_2 = start_point(start_in=0.5, p=1, k=3, jump=1)
#
# print "number of features:",layer2
# print "jump:", jump2
# print "the receptive filed size is:", rf_2
# print "the starting point:", start_2

from keras.backend import set_image_dim_ordering
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, Dropout, Activation, Reshape, Dense, ZeroPadding2D
from keras.layers import merge
from keras.models import Sequential


set_image_dim_ordering("tf")

input_shape = (5,5,1)
inputs = Input(shape=input_shape)
conv1 = Convolution2D(nb_filter=1, nb_row=3, nb_col=3, border_mode='same', subsample=(2,2),
                        activation ="relu", input_shape=input_shape, name="conv1")(inputs)   # subsample = strides

# conv1_shape =  conv1.get_shape()
# print conv1_shape[1]
# print conv1_shape[2]


print "-------------------"


# layer1 = nb_out_features(nb_features=input_shape[0], p=1, k=3, s=2)
# jump1 = distance_2pixel(jump_in=1, s=2)
# rf_1 = receptive_field_size(r_in=1, k=3, jump=1)
# start_1 = start_point(start_in=0.5, p=1, k=3, jump=1)


conv2 = Convolution2D(nb_filter=1, nb_row=3, nb_col=3, border_mode='same', subsample=(2,2),
                        activation ="relu", input_shape=(5,5,1), name="conv2")(conv1)   # subsample = strides

# conv2_shape =  conv2.get_shape()
# print conv2_shape[1]
# print conv2_shape[2]



model = Model(inputs,conv2)
#summary =  model.summary()

configuration = model.get_config()

# print configuration
# print type(configuration)
print configuration.keys()

print "-----------------------------------------------------------------------------------"


layers = configuration["layers"]
# print type(layers)
# print len(layers)
#
# print layers[0].keys()
# print layers[0]["class_name"]
# print layers[0]["config"]
#
# print layers[0]["config"].keys()


layer_information = layers[1]["config"]
keys = layers[1]["config"].keys()

print layer_information["nb_col"]
print layer_information["nb_row"]
print layer_information["subsample"][0]
print layer_information["border_mode"]
print layer_information["batch_input_shape"]



# for i in range(len(layers)):
#     print layers[i]
#     print "type der layers", type(layers[i])
#     print layers[i].keys()


    # print "class_name:", layers[i]["class_name"]
    # print "config:", layers[i]["config"]
    # layer_infotmation = layers[i]["config"]
    # keys = layer_infotmation.keys()
    #
    # print "-----------------------------------------------------------------------------------"
    # for j in range((len(keys))):
    #
    #     print keys[j], "is", layer_infotmation[keys[j]]
    #
    #
    #
    #
    #
    #
    # #print "inbound_nodes:", layers[i]["inbound_nodes"]
    # #print "name:", layers[i]["name"]
    # print "------------------------------------------------------------------------------"


#
# print configuration["input_layers"]
# print type(configuration["input_layers"])
#
# print configuration["output_layers"]
# print type(configuration["output_layers"])
#
# print configuration["name"]
# print type(configuration["name"])
