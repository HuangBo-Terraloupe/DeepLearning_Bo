""" Resnet based Pyramid scale pooling network for semantic segmentation

Example:
    # Required input_shape is (240, 240)
     model = pspnet.construct((240,240), batch_size=2, n_labels=5, n_channels=3)

"""

from __future__ import print_function

import keras.backend as backend
from keras.layers import Activation, Reshape, Permute
from keras.layers import BatchNormalization, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, UpSampling2D
from keras.layers import merge, Input
from keras.models import Model

from objectdetection.utils import logger
from resnet import conv_block, identity_block


def construct(input_shape, n_labels, n_channels=3, batch_size=None, weights_path=None):
    """ Constructs 50 layers PSPNet with pyramid scale pooling

    Args:
        input_shape (tuple): A tuple containing (height, width)
        n_labels (int): Number of input labels
        n_channels (int): Number of input channels
        batch_size (int): Batch size for the model
        weights_path (str, optional): Path to a pretrained model

    Returns: A Keras model

    """
    if backend.image_dim_ordering() == 'tf':
        bn_axis = 3
        batch_shape = (batch_size, input_shape[0], input_shape[1], n_channels)
    else:
        batch_shape = (batch_size, n_channels, input_shape[0], input_shape[1])
        bn_axis = 1

    img_input = Input(batch_shape=batch_shape)

    x = ZeroPadding2D((1, 1))(img_input)
    x = Convolution2D(64, 3, 3, subsample=(1, 1), name='conv1_1', bias=False, init='he_normal')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1_1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, subsample=(1, 1), name='conv1_2', bias=False, init='he_normal')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1_2')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(128, 3, 3, subsample=(1, 1), name='conv1_3', bias=False, init='he_normal')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1_3')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Dilation starts here and no more pooling in conv_block
    # Dilation rate = 2
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(1, 1), dilation=(2, 2))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dilation=(2, 2))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dilation=(2, 2))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dilation=(2, 2))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dilation=(2, 2))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dilation=(2, 2))

    # Dilation rate = 4
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1), dilation=(4, 4))
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation=(4, 4))
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation=(4, 4))

    # Pyramid Scale Pooling (PSP)
    # Pooling Size: 10
    x_1 = AveragePooling2D((10, 10), name='avg_pool_1')(x)
    x_1 = Convolution2D(512, 1, 1, subsample=(1, 1), name='conv6_1', bias=False, init='he_normal')(x_1)
    x_1 = BatchNormalization(axis=bn_axis, name='bn_conv6_1')(x_1)
    x_1 = Activation('relu')(x_1)
    x_1 = UpSampling2D(size=(10, 10))(x_1)

    # Pooling Size: 20
    x_2 = AveragePooling2D((20, 20), name='avg_pool_2')(x)
    x_2 = Convolution2D(512, 1, 1, subsample=(1, 1), name='conv6_2', bias=False, init='he_normal')(x_2)
    x_2 = BatchNormalization(axis=bn_axis, name='bn_conv6_2')(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = UpSampling2D(size=(20, 20))(x_2)

    # Pooling Size: 30
    x_3 = AveragePooling2D((30, 30), name='avg_pool_3')(x)
    x_3 = Convolution2D(512, 1, 1, subsample=(1, 1), name='conv6_3', bias=False, init='he_normal')(x_3)
    x_3 = BatchNormalization(axis=bn_axis, name='bn_conv6_3')(x_3)
    x_3 = Activation('relu')(x_3)
    x_3 = UpSampling2D(size=(30, 30))(x_3)

    # Pooling Size: 60
    x_4 = AveragePooling2D((60, 60), name='avg_pool_4')(x)
    x_4 = Convolution2D(512, 1, 1, subsample=(1, 1), name='conv6_4', bias=False, init='he_normal')(x_4)
    x_4 = BatchNormalization(axis=bn_axis, name='bn_conv6_4')(x_4)
    x_4 = Activation('relu')(x_4)
    x_4 = UpSampling2D(size=(60, 60))(x_4)

    # Feature Aggregation
    x = merge([x_1, x_2, x_3, x_4, x], mode='concat', concat_axis=bn_axis)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(512, 3, 3, subsample=(1, 1), name='conv7', bias=False, init='he_normal')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv7')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # Fully convolutional layer
    x = Convolution2D(n_labels, 1, 1, subsample=(1, 1), name='fc8', bias=False, init='he_normal')(x)
    x = UpSampling2D(size=(4, 4))(x)

    if backend.image_dim_ordering() == 'th':
        x = Permute(dims=(2, 3, 1))(x)

    x = Reshape(target_shape=(input_shape[0]*input_shape[1], n_labels))(x)
    x = Activation(activation='softmax', name='fc4')(x)

    model = Model(img_input, x)

    if weights_path:
        logger.info("Loading pretrained weights from '%s'" % weights_path)
        model.load_weights(weights_path)

    return model
