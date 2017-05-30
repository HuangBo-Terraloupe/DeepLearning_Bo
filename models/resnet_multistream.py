#from __future__ import print_function

import keras.backend as backend
from keras.layers import Activation, Reshape
from keras.layers import BatchNormalization
from keras.layers import Convolution2D, ZeroPadding2D, Deconvolution2D
from keras.layers import merge, Input
from keras.models import Model
from keras.backend import set_image_dim_ordering

#from objectdetection.utils import logger
from resnet import conv_block, identity_block

def construct(input_shape, n_labels, n_channels=3, batch_size=None): #, weights_path=None):
    """ Constructs 50 layers resnet model with multi-stream feature aggregation.
        Model has a 496X496 pixel field of view

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

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x_64 = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x_64, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x_32 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x_32, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x_16 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    fc_64 = Convolution2D(n_labels, 1, 1, subsample=(1, 1), name='fc_64', bias=False, init='he_normal')(x_64)
    fc_32 = Convolution2D(n_labels, 1, 1, subsample=(1, 1), name='fc_32', bias=False, init='he_normal')(x_32)
    fc_16 = Convolution2D(n_labels, 1, 1, subsample=(1, 1), name='fc_16', bias=False, init='he_normal')(x_16)

    up_32 = Deconvolution2D(n_labels, 2, 2, subsample=(2,2), output_shape=(batch_size, 124, 124, n_labels))(fc_32)
    up_16 = Deconvolution2D(n_labels, 4, 4, subsample=(4, 4), output_shape=(batch_size, 124, 124, n_labels))(fc_16)

    aggr = merge([fc_64, up_32, up_16], mode='sum', concat_axis=bn_axis)

    score = Deconvolution2D(n_labels, 4, 4, subsample=(4, 4),
                            output_shape=(batch_size, input_shape[0], input_shape[1], n_labels))(aggr)

    x = Reshape(target_shape=(input_shape[0]*input_shape[1], n_labels))(score)
    x = Activation(activation='softmax', name='softmax')(x)

    model = Model(img_input, x)

    # if weights_path:
    #     logger.info("Loading pretrained weights from '%s'" % weights_path)
    #     model.load_weights(weights_path)

    return model


if __name__ == '__main__':

    set_image_dim_ordering('tf')
    block_size = 496
    model = construct(input_shape = (block_size, block_size), n_labels=11, n_channels=3, batch_size=1)
    print model.summary()
    n =0
    for layer in model.layers:
        n = n+1
        layer_configuration = layer.get_config()

    print n