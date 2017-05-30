import keras

def receptive_field(model, stop_flag=None, single_layer_name=None):
    '''
    The receptive field is defined as the region in the input space that a particular CNN's feature is looking at.
    A receptive field of a feature can be fully described by its center location and its size.

    This arithmetic can directly work on the sequential model that is a stack of Conv and Pooling layers, 
    If in the model exist shared layers (Multi-input and multi-output), a stop_flag need to be provided. Otherwise the 
    calculation will be stopped at any upsample operation(upsampling and deconvolution). 

    When deal with the model like resnet, to keep this arithmetic valid, we ignore the 1x1 convolution in the 
    conv_block, that means we consider the whole block as one layer, and choose the maximal receptive field of the 
    layers inside the block as the receptive field of whole block.

    For more information: 
    <https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional
    -neural-networks-e0f514068807>

    :param model: a keras model            
    :param stop_flag: the name of layer, where we stop the calculation
    :param single_layer_name: we just look at this single layer, if it's None, this function will print all layers
    :return: receptive fields of one single layer or all layers 

    example:
            1. unet
            detector = unet.UNet(batch_size=1, input_shape=(252, 252), n_channels=3, no_classes=10, weight_file=None)
            model = detector.build_model()
            receptive_fields(model, stop_flag=None, single_layer_name="convolution2d_8")
            receptive_fields(model, stop_flag=None, single_layer_name=None)
            2. resnet
            model = resnet_multistream.construct(input_shape= (496, 496), n_labels = 11, n_channels=3, batch_size=1)
            receptive_fields(model, stop_flag="res5c_branch2c")
            receptive_fields(model, stop_flag="res5c_branch2c", single_layer_name="res5c_branch2c")
            3. pspnet
            model = pspnet.construct(input_shape= (240, 240), n_labels = 11, n_channels=3, batch_size=1)
            receptive_fields(model, stop_flag="res5c_branch2c")
            receptive_fields(model, stop_flag="res5c_branch2c", single_layer_name="res5c_branch2c")
    '''

    def calculate_padding(feature_input, feature_output, strides, kernel):
        return (feature_output - 1) * strides - feature_input + kernel

    def printLayer(layer, layer_name):
        print(layer_name + ":")
        print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \n \t start: %s " % (
            layer[0], layer[1], layer[2], layer[3]))

    def single_layer_receptive_field(receptive_in, layer_info):

        jump = receptive_in[1]
        receptive_file = receptive_in[2]
        start_point = receptive_in[3]

        kernel = layer_info[0]
        strides = layer_info[1]
        padding = layer_info[2]
        nb_features_nest = layer_info[3]

        jump_next = jump * strides
        receptive_file_next = receptive_file + (kernel - 1) * jump
        start_point_next = start_point + ((kernel - 1) / 2 - padding) * jump

        receptive_field = (nb_features_nest, jump_next, receptive_file_next, start_point_next)

        return receptive_field

    # check the input
    input_layer = model.input_shape
    assert len(input_layer) == 4 and input_layer[1] == input_layer[2], \
        "the dimension of the input should be 4, and the image need to be squre"

    # record
    all_layer_names = []
    layer_Infos = []

    receptive_in = (input_layer[1], 1, 1, 0.5)
    for layer in model.layers:
        layer_configuration = layer.get_config()

        if type(layer) is keras.layers.convolutional.Convolution2D:

            # ignore the 1x1 convolution in conv_block
            if str(layer_configuration["name"])[-1] == "1" and str(layer_configuration["name"])[0:3] == "res":
                continue

            name = layer_configuration["name"]
            nb_features_next = layer.output_shape[1]
            kernel = layer_configuration["nb_col"]
            strides = layer_configuration["subsample"][0]
            padding = calculate_padding(layer.input_shape[1],
                                        layer.output_shape[1],
                                        layer_configuration["subsample"][0],
                                        layer_configuration["nb_col"]
                                        )
            layer_info = (kernel, strides, padding, nb_features_next)
            receptive_in = single_layer_receptive_field(receptive_in, layer_info)
            all_layer_names.append(name)
            layer_Infos.append(receptive_in)

        elif type(layer) is keras.layers.convolutional.AtrousConvolution2D:

            name = layer_configuration["name"]
            nb_features_next = layer.output_shape[1]
            kernel = layer_configuration["nb_col"]
            dilation = layer_configuration["atrous_rate"][0]
            actual_kernel = kernel + (kernel - 1) * (dilation - 1)
            strides = layer_configuration["subsample"][0]
            padding = calculate_padding(layer.input_shape[1],
                                        layer.output_shape[1],
                                        layer_configuration["subsample"][0],
                                        actual_kernel
                                        )
            layer_info = (actual_kernel, strides, padding, nb_features_next)
            receptive_in = single_layer_receptive_field(receptive_in, layer_info)
            all_layer_names.append(name)
            layer_Infos.append(receptive_in)

        elif type(layer) in {keras.layers.pooling.MaxPooling2D,
                             keras.layers.pooling.AveragePooling2D,
                             keras.layers.pooling.GlobalMaxPooling2D
                             }:

            name = layer_configuration["name"]
            nb_features_next = layer.output_shape[1]
            kernel = layer_configuration["pool_size"][0]
            strides = layer_configuration["strides"][0]
            padding = calculate_padding(layer.input_shape[1],
                                        layer.output_shape[1],
                                        layer_configuration["strides"][0],
                                        layer_configuration["pool_size"][0]
                                        )
            layer_info = (kernel, strides, padding, nb_features_next)
            receptive_in = single_layer_receptive_field(receptive_in, layer_info)
            all_layer_names.append(name)
            layer_Infos.append(receptive_in)

        # stop the calculation at the given layer
        if layer_configuration["name"] == stop_flag:
            break
        else:
            continue

    # just print single layer
    if single_layer_name:
        layer_idx = all_layer_names.index(single_layer_name)
        nb_features = layer_Infos[layer_idx][0]
        jump = layer_Infos[layer_idx][1]
        receptive_fields_size = layer_Infos[layer_idx][2]
        start = layer_Infos[layer_idx][3]
        printLayer((nb_features, jump, receptive_fields_size, start), single_layer_name)

        idx_x = int(raw_input("index of the feature in x dimension (from 0 to" + " " + str(nb_features - 1) + ")"))
        idx_y = int(raw_input("index of the feature in y dimension (from 0 to" + " " + str(nb_features - 1) + ")"))
        assert (idx_x < nb_features), "the index must smaller than number of features in this layer"
        assert (idx_y < nb_features), "the index must smaller than number of features in this layer"
        print ("receptive field: (%s, %s)" % (receptive_fields_size, receptive_fields_size))
        print ("center: (%s, %s)" % (start + idx_x * jump, start + idx_y * jump))

    # print all layers
    else:
        receptive_field_input = (input_layer[1], 1, 1, 0.5)
        printLayer(receptive_field_input, "input_layer")
        for i in range(len(layer_Infos)):
            nb_features = layer_Infos[i][0]
            jump = layer_Infos[i][1]
            receptive_fields_size = layer_Infos[i][2]
            start = layer_Infos[i][3]
            printLayer((nb_features, jump, receptive_fields_size, start), all_layer_names[i])


