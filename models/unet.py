import pickle
from objectdetection.dataset.image_dataset import ImageDataset
from objectdetection import callbacks
import numpy as np
import numpy

from keras.optimizers import Adam
from keras.backend import set_image_dim_ordering
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, Dropout, Activation, Reshape
from keras.layers import merge
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import (ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard)

from objectdetection.utils import logger


class UNet(object):
    '''
    UNet implementation with transposed convolutions. The input size to a unet should be multiple of
    32x+220 where x is in N. This implementation is slightly modified from original paper and outputs
    same dimensional response maps as input.

    exsample:
    segmentation = UNet(batch_size=batch_size, input_shape=(block_size, block_size), n_channels=3, no_classes=no_classes, weight_file=None)
    model = segmentation.build_model()
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
        up_conv = Deconvolution2D(no_features, 5, 5, output_shape=out_shape, subsample=(2, 2))(input)
        # up_conv = Convolution2D(no_features, 2, 2)(UpSampling2D()(input))
        merged = merge([concat_tensor, up_conv], mode='concat', concat_axis=3)
        return merged

    def conv3_3(self, input, no_features):
        conv1 = Convolution2D(no_features, 3, 3, activation='relu')(input)
        conv2 = Convolution2D(no_features, 3, 3, activation='relu')(conv1)
        return conv2

    def build_model(self):
        input = Input(batch_shape=self.input_shape, name='input_1')
        conv1_1 = Convolution2D(64, 3, 3, activation='relu')(input)
        conv1_2 = Convolution2D(64, 3, 3, activation='relu')(conv1_1)
        conv1_out = MaxPooling2D()(conv1_2)

        conv2_1 = Convolution2D(128, 3, 3, activation='relu')(conv1_out)
        dropout2 = Dropout(0.5)(conv2_1)
        conv2_2 = Convolution2D(128, 3, 3, activation='relu')(dropout2)
        conv2_out = MaxPooling2D()(conv2_2)

        conv3_1 = Convolution2D(256, 3, 3, activation='relu')(conv2_out)
        dropout3 = Dropout(0.5)(conv3_1)
        conv3_2 = Convolution2D(256, 3, 3, activation='relu')(dropout3)
        conv3_out = MaxPooling2D()(conv3_2)

        conv4_1 = Convolution2D(512, 3, 3, activation='relu')(conv3_out)
        dropout4 = Dropout(0.5)(conv4_1)
        conv4_2 = Convolution2D(512, 3, 3, activation='relu')(dropout4)
        conv4_out = MaxPooling2D()(conv4_2)

        conv5_1 = Convolution2D(1024, 3, 3, activation='relu')(conv4_out)
        dropout5 = Dropout(0.5)(conv5_1)
        conv5_2 = Convolution2D(1024, 3, 3, activation='relu')(dropout5)
        conv5_out = MaxPooling2D()(conv5_2)

        up_conv1 = self.upconv2_2(conv5_out, conv4_out, 512)
        # conv6_out = self.conv3_3(up_conv1, 512)

        up_conv2 = self.upconv2_2(up_conv1, conv3_out, 256)
        # conv7_out = self.conv3_3(up_conv2, 256)

        up_conv3 = self.upconv2_2(up_conv2, conv2_out, 128)
        # conv8_out = self.conv3_3(up_conv3, 128)

        up_conv4 = self.upconv2_2(up_conv3, conv1_out, 64)
        # conv9_out = self.conv3_3(up_conv4, 64)

        out_shape = [dim.value for dim in input.get_shape()]
        out_shape = [self.batch_size] + out_shape[1:3] + [self.no_classes]
        output = Deconvolution2D(self.no_classes, 5, 5, out_shape, subsample=(2, 2))(up_conv4)
        output = Reshape((self.input_shape[1] * self.input_shape[2], self.no_classes))(output)
        output = Activation(activation='softmax', name='class_out')(output)

        model = Model(input, output)
        if self.weight_file:
            logger.info('Loading weights from file:{}'.format(self.weight_file))
            model.load_weights(self.weight_file)
        return model


def construct(input_shape, n_labels, n_channels=3, batch_size=None, weights_path=None):
    model = UNet(batch_size=batch_size, input_shape=input_shape, n_channels=n_channels,
                 no_classes=n_labels, weight_file=weights_path).build_model()
    return model

