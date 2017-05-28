import keras
from models import unet, fcn8_vgg

class Receptive_field:

    def __init__(self, model):
        self.model = model

    def calculate_padding(self, feature_input, feature_output, strides, kernel):
        return (feature_output - 1) * strides - feature_input + kernel

    def printLayer(self ,layer, layer_name):
        print(layer_name + ":")
        print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \n \t start: %s " % (
        layer[0], layer[1], layer[2], layer[3]))

    def single_layer_receptive_field(self, receptive_in, layer_info):

        jump = receptive_in[1]
        receptive_file = receptive_in[2]
        start_point = receptive_in[3]

        kernel = layer_info[0]
        strides = layer_info[1]
        padding = layer_info[2]
        nb_features_nest = layer_info[3]

        jump_next = jump*strides
        receptive_file_next = receptive_file + (kernel - 1) * jump
        start_point_next = start_point + ((kernel - 1) / 2 - padding) * jump

        receptive_field = (nb_features_nest, jump_next, receptive_file_next, start_point_next)

        return receptive_field


    def receptive_fileds(self):

        # check the input
        input_layer = self.model.input_shape
        if len(input_layer) == 4 and input_layer[1]==input_layer[2]:
            image_size = input_layer[1]
        else:
            print "the dimension of the input should be 4, and the image need to be squre"

        # record
        layer_names = []
        layer_Infos = []

        last_layer_size = image_size
        receptive_in = (image_size, 1, 1, 0.5)
        self.printLayer(receptive_in, "input_layer")

        for layer in self.model.layers:
            layer_configuration = layer.get_config()

            # we calculate the receptive fieds to track the information from every layer until any upsample operation
            if type(layer) in {keras.layers.convolutional.UpSampling2D, keras.layers.convolutional.Deconvolution2D}:
                print "stop at upsample opration"
                break

            # check if we missing some layers, which changes the size of feature maps
            if last_layer_size == layer.input_shape[1]:
                pass
            else:
                print "the name of layer", layer_configuration["name"]
                print "Error: the input shape of current layer is not equal to the output shape of last layer"
                break


            # calculate the receptive fields for convolution operation
            if  type(layer) is keras.layers.convolutional.Convolution2D:

                name = layer_configuration["name"]
                nb_features_next = layer.output_shape[1]
                kernel = layer_configuration["nb_col"]
                strides = layer_configuration["subsample"][0]
                padding = self.calculate_padding(layer.input_shape[1],
                                                 layer.output_shape[1],
                                                 layer_configuration["subsample"][0],
                                                 layer_configuration["nb_col"]
                                                 )

                layer_info = (kernel, strides, padding, nb_features_next)
                receptive_in = self.single_layer_receptive_field(receptive_in, layer_info)

                layer_names.append(name)
                layer_Infos.append(receptive_in)
                last_layer_size = layer.output_shape[1]

                self.printLayer(receptive_in, name)

            elif type(layer) is keras.layers.convolutional.AtrousConvolution2D:

                name = layer_configuration["name"]
                nb_features_next = layer.output_shape[1]
                kernel = layer_configuration["nb_col"]
                dilation = layer_configuration["atrous_rate"][0]
                actual_kernel = kernel + (kernel-1)*(dilation-1)
                strides = layer_configuration["subsample"][0]
                padding = self.calculate_padding(layer.input_shape[1],
                                                 layer.output_shape[1],
                                                 layer_configuration["subsample"][0],
                                                 actual_kernel
                                                 )

                layer_info = (actual_kernel, strides, padding, nb_features_next)
                receptive_in = self.single_layer_receptive_field(receptive_in, layer_info)

                layer_names.append(name)
                layer_Infos.append(receptive_in)
                last_layer_size = layer.output_shape[1]

                self.printLayer(receptive_in, name)



                # print "layer name is", layer_configuration["name"]
                # print "the input shape", layer.input_shape
                # print "the output shape", layer.output_shape
                # print "strides", layer_configuration["subsample"][0]
                # print "kernel size is", layer_configuration["nb_col"]
                # print "padding", calculate_padding(layer.input_shape[1],layer.output_shape[1],
                #                                    layer_configuration["subsample"][0],layer_configuration["nb_col"]
                #                                    )
                # print '----------------------------------------------------'


            # elif type(layer) is keras.layers.pooling.AveragePooling2D:
            #      print layer_configuration.keys()
            #
            #      print "layer name is", layer_configuration["name"]
            #      print "the input shape", layer.input_shape
            #      print "the output shape", layer.output_shape
            #      print "strides", layer_configuration["strides"][0]
            #      print "kernel size is", layer_configuration["pool_size"][0]

            elif type(layer) in {keras.layers.pooling.MaxPooling2D,
                                 keras.layers.pooling.AveragePooling2D,
                                 keras.layers.pooling.GlobalMaxPooling2D}:

                name = layer_configuration["name"]
                nb_features_next = layer.output_shape[1]
                kernel = layer_configuration["pool_size"][0]
                strides = layer_configuration["strides"][0]
                padding = self.calculate_padding(layer.input_shape[1],
                                                 layer.output_shape[1],
                                                 layer_configuration["strides"][0],
                                                 layer_configuration["pool_size"][0]
                                                 )

                layer_info = (kernel, strides, padding, nb_features_next)
                receptive_in = self.single_layer_receptive_field(receptive_in, layer_info)

                layer_names.append(name)
                layer_Infos.append(receptive_in)
                last_layer_size = layer.output_shape[1]

                self.printLayer(receptive_in, name)

                # print "layer name is", layer_configuration["name"]
                # print "the input shape", layer.input_shape
                # print "the output shape", layer.output_shape
                # print "strides", layer_configuration["strides"][0]
                # print "kernel size is", layer_configuration["pool_size"][0]
                # print "padding", calculate_padding(layer.input_shape[1],layer.output_shape[1],
                #                                    layer_configuration["strides"][0],layer_configuration["pool_size"][0]
                #                                    )
                # print '----------------------------------------------------'

            # for the operation Zero-padding or crop, which does not change the receptive fields but change the image shape
            else:
                last_layer_size = layer.output_shape[1]


        layer_name = raw_input("Layer name where the feature in: ")
        layer_idx = layer_names.index(layer_name)

        nb_features = layer_Infos[layer_idx][0]
        jump = layer_Infos[layer_idx][1]
        receptive_fileds_size = layer_Infos[layer_idx][2]
        start = layer_Infos[layer_idx][3]

        idx_x = int(
            raw_input("index of the feature in x dimension (from 0 to" + " " + str(nb_features - 1) + ")"))
        idx_y = int(
            raw_input("index of the feature in y dimension (from 0 to" + " " + str(nb_features - 1) + ")"))

        assert (idx_x < nb_features), "the index must smaller than number of features in this layer"
        assert (idx_y < nb_features), "the index must smaller than number of features in this layer"

        print ("receptive field: (%s, %s)" % (receptive_fileds_size, receptive_fileds_size))
        print ("center: (%s, %s)" % (start + idx_x * jump, start + idx_y * jump))

    # def receptive_field_one_feature(self, layer_names, layer_Infos):
    #     layer_name = raw_input("Layer name where the feature in: ")
    #     layer_idx = layer_names.index(layer_name)
    #
    #     nb_features = layer_Infos[layer_idx][0]
    #     jump = layer_Infos[layer_idx][1]
    #     receptive_fileds_size = layer_Infos[layer_idx][2]
    #     start = layer_Infos[layer_idx][3]
    #
    #     idx_x = int(raw_input("index of the feature in x dimension (from 0 to" + " " + str(nb_features - 1) + ")"))
    #     idx_y = int(raw_input("index of the feature in y dimension (from 0 to" + " " + str(nb_features - 1) + ")"))
    #
    #
    #     assert (idx_x < nb_features),"the index must smaller than number of features in this layer"
    #     assert (idx_y < nb_features),"the index must smaller than number of features in this layer"
    #
    #     print ("receptive field: (%s, %s)" % (receptive_fileds_size, receptive_fileds_size))
    #     print ("center: (%s, %s)" % (start + idx_x * jump, start + idx_y * jump))
    #
    #     return None
    #
    # def run(self):
    #     layer_names, layer_Infos = self.receptive_fileds()
    #     self.receptive_field_one_feature(layer_names, layer_Infos)
    #     return None


model = fcn8_vgg.Fcn_8(batch_size=1, input_shape=(480,480), n_channels=3, no_classes=11)
model = model.build_model()
# print model.summary()


# detector = unet.UNet(batch_size=1, input_shape=(252, 252), n_channels = 3, no_classes=10, weight_file=None)
# model = detector.build_model()
# print model.summary()


rf = Receptive_field(model)
rf.receptive_fileds()

