"""
Implementaiton of Resnet building blocks, which can be reused in different resnet archtectures
"""

from __future__ import print_function

from keras.layers import merge
from keras.layers import Activation
from keras.layers import Convolution2D, AtrousConvolution2D
from keras.layers import BatchNormalization
import keras.backend as backend


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation=(1, 1)):
    """ The identity_block is the block that has no conv layer at shortcut

    Args:
        input_tensor: input tensor
        kernel_size (int): defualt 3, the kernel size of middle conv layer at main path
        filters (list): list of integers, the nb_filters of 3 conv layer at main path
        stage (int): current stage label, used for generating layer names
        block (char): 'a','b'..., current block label, used for generating layer names
        dilation (tuple, optional): Dilation rate, only applicable for Atrous convolution. Default (1,1)

    Returns: An Identity resnet block

    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    if backend.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a', bias=False, init='he_normal')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = AtrousConvolution2D(nb_filter2, kernel_size, kernel_size, bias=False,
                      border_mode='same', name=conv_name_base + '2b', atrous_rate=dilation, init='he_normal')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False, init='he_normal')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dilation=(1, 1)):

    # conv_block is the block that has a conv layer at shortcut
    """

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    if backend.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a', bias=False, init='he_normal')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = AtrousConvolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b', atrous_rate=dilation, bias=False, init='he_normal')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False, init='he_normal')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1', bias=False, init='he_normal')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)


    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x
