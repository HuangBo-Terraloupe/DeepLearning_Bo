from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, Dropout, Activation, Reshape
from keras.layers import merge



class Fcn_8(object):
    '''
    exsample:
    model = Fcn_8(batch_size=batch_size, input_shape=(block_size,block_size), n_channels=3, no_classes=11)
    model = model.build_model()
    block_size = 240
    '''

    def __init__(self, batch_size, input_shape, n_channels, no_classes, weight_file=None):
        self.batch_size = batch_size
        self.patch_size = input_shape[0], input_shape[1]
        self.input_channels = n_channels
        self.input_shape = (self.batch_size,) + self.patch_size + (self.input_channels,)
        self.out_channels = no_classes
        self.output_shape = [self.batch_size, input_shape[0], input_shape[1], self.out_channels]
        self.no_classes = no_classes
        self.weight_file = weight_file

    def upconv2_2(self, input, concat_tensor, no_features):
        out_shape = [dim.value for dim in concat_tensor.get_shape()]
        up_conv = Deconvolution2D(no_features, 4, 4, output_shape=out_shape, subsample=(2, 2))(input)
        # up_conv = Convolution2D(no_features, 2, 2)(UpSampling2D()(input))
        merged = merge([concat_tensor, up_conv], mode='concat', concat_axis=3)
        return merged

    def build_model(self):
        input = Input(batch_shape=self.input_shape, name='input_1')

        fileter_size = 3

        # Block 1
        conv1_1 = Convolution2D(64, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv1_1')(
            input)
        conv1_2 = Convolution2D(64, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv1_2')(
            conv1_1)
        conv1_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same', name='conv1_out')(conv1_2)

        # Block 2
        conv2_1 = Convolution2D(128, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv2_1')(
            conv1_out)
        conv2_2 = Convolution2D(128, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv2_2')(
            conv2_1)
        conv2_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same', name='conv2_out')(conv2_2)

        # Block 3
        conv3_1 = Convolution2D(256, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv3_1')(
            conv2_out)
        conv3_2 = Convolution2D(256, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv3_2')(
            conv3_1)
        conv3_3 = Convolution2D(256, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv3_3')(
            conv3_2)
        conv3_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same', name='conv3_out')(conv3_3)

        # Block 4
        conv4_1 = Convolution2D(512, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv4_1')(
            conv3_out)
        conv4_2 = Convolution2D(512, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv4_2')(
            conv4_1)
        conv4_3 = Convolution2D(512, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv4_3')(
            conv4_2)
        conv4_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same', name='conv4_out')(conv4_3)

        # Block 5
        conv5_1 = Convolution2D(512, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv5_1')(
            conv4_out)
        conv5_2 = Convolution2D(512, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv5_2')(
            conv5_1)
        conv5_3 = Convolution2D(512, fileter_size, fileter_size, activation='relu', border_mode='same', name='conv5_3')(
            conv5_2)
        conv5_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same', name='conv5_out')(conv5_3)

        # Block 6
        conv6_1 = Convolution2D(1024, 7, 7, activation='relu', border_mode='same', name='conv6_1')(conv5_out)
        conv6_out = Dropout(0.5)(conv6_1)

        # Block 7
        conv7_1 = Convolution2D(1024, 1, 1, activation='relu', border_mode='same', name='conv7_1')(conv6_out)
        conv7_out = Dropout(0.5)(conv7_1)

        # De1
        score_conv7_out = Convolution2D(self.no_classes, 1, 1, border_mode='same')(conv7_out)
        score_pool4 = Convolution2D(self.no_classes, 1, 1, border_mode='same')(conv4_out)

        out_shape = [dim.value for dim in score_pool4.get_shape()]
        up_conv_1 = Deconvolution2D(self.no_classes, 4, 4, output_shape=out_shape, border_mode="same",
                                    subsample=(2, 2))(score_conv7_out)
        upscore_1 = merge([score_pool4, up_conv_1], mode='sum', concat_axis=-1)

        # De2
        score_pool3 = Convolution2D(self.no_classes, 1, 1, border_mode='same')(conv3_out)
        out_shape = [dim.value for dim in score_pool3.get_shape()]
        up_conv_2 = Deconvolution2D(self.no_classes, 4, 4, output_shape=out_shape, border_mode="same",
                                    subsample=(2, 2))(upscore_1)
        upscore_2 = merge([score_pool3, up_conv_2], mode='sum', concat_axis=-1)

        # up_conv1 = self.upconv2_2(conv7_out, conv4_out, 512)
        # conv6_out = self.convfileter_size_fileter_size(up_conv1, 512)

        # up_conv2 = self.upconv2_2(up_conv1, conv3_out, 256)
        # conv7_out = self.convfileter_size_fileter_size(up_conv2, 256)

        out_shape = [dim.value for dim in input.get_shape()]
        out_shape = [self.batch_size] + out_shape[1:fileter_size] + [self.no_classes]
        output = Deconvolution2D(self.no_classes, 16, 16, out_shape, border_mode="same", subsample=(8, 8))(upscore_2)
        output = Reshape((self.input_shape[1] * self.input_shape[2], self.no_classes))(output)
        output = Activation(activation='softmax', name='class_out')(output)

        model = Model(input, output)

        return model

