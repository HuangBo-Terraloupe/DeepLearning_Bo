from keras.applications import InceptionV3



from keras.layers import Conv2D, MaxPooling2D, Input, concatenate

def inception(input,
              filters_1x1,
              filters_3x3_reduce, filters_3x3,
              filters_5x5_reduce, filters_5x5,
              filters_pool_proj):
    """

    :param input:
    :param filters_1x1:
    :param filters_3x3_reduce:
    :param filters_3x3:
    :param filters_5x5_reduce:
    :param filters_5x5:
    :param filters_pool_proj:
    :return:
    """
    same = 'same'
    relu = 'relu'

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding=same, activation=relu)(input)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding=same, activation=relu)(input)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding=same, activation=relu)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding=same, activation=relu)(input)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding=same, activation=relu)(conv_5x5)

    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding=same)(input)
    maxpool_proj = Conv2D(filters_pool_proj, (1, 1), padding=same, activation=relu)(maxpool)

    # concatenate by channels
    # axis=3 or axis=-1(default) for tf backend
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)

    return output


from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Reshape, Activation, GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
