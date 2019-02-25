from keras.applications.xception import Xception
from keras.layers import Reshape, Activation, Conv2DTranspose, BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.models import Model

# Usage information:
# Native resolution for Xception was 299x299
# Now supported resolutions are 256 +- a multiple of 32
# Tested resolutions to determine max batch size (only potentials of 2 are tested) on a 8GB GPU are
# 224x224: 16 (freezed base) /  8 (fully trainable) 28504259 (trainable on freeze base) / 42566555 (on fully trainable)
# 256x256: 16 (freezed base) /  8 (fully trainable) 28504259 (trainable on freeze base) / 42566555 (on fully trainable)
# 288x288: 16 (freezed base) /  8 (fully trainable) 28504259 (trainable on freeze base) / 42566555 (on fully trainable)
# 320x320:  8 (freezed base) /  4 (fully trainable) 28504259 (trainable on freeze base) / 42566555 (on fully trainable)


def construct(input_shape, n_labels, n_channels=3, batch_size=None, freeze=False, weights_path=None):
    # create the base pre-trained model
    if n_channels == 3:
        base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape+(n_channels,))
    elif n_channels == 4:
        base_model = Xception(weights=None, include_top=False, input_shape=input_shape + (n_channels,))
    else:
        assert('Unsupported number of channels !')

    x = Conv2DTranspose(1024, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(base_model.layers[-17].output)

    x = Conv2DTranspose(1024, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True,
                           beta_initializer="zeros", gamma_initializer="ones",
                           moving_mean_initializer="zeros", moving_variance_initializer="ones",
                           beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(x)

    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2), data_format=None)(x)

    x = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = Conv2DTranspose(512, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True,
                           beta_initializer="zeros", gamma_initializer="ones",
                           moving_mean_initializer="zeros", moving_variance_initializer="ones",
                           beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(x)

    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2), data_format=None)(x)

    x = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True,
                           beta_initializer="zeros", gamma_initializer="ones",
                           moving_mean_initializer="zeros", moving_variance_initializer="ones",
                           beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(x)

    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2), data_format=None)(x)

    x = Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True,
                           beta_initializer="zeros", gamma_initializer="ones",
                           moving_mean_initializer="zeros", moving_variance_initializer="ones",
                           beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(x)

    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2), data_format=None)(x)

    x = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = Conv2DTranspose(n_labels, (3, 3), strides=(1, 1), padding='same',
                        data_format=None, activation='relu',
                        use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', kernel_regularizer=None,
                        bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None)(x)

    x = Reshape(target_shape=(input_shape[0] * input_shape[1], n_labels))(x)

    x = Activation(activation='softmax', name='softmax')(x)

    model = Model(input=base_model.input, output=x)
    print(model.summary())

    for layer in base_model.layers:
        layer.trainable = not freeze

    if weights_path:
        model.load_weights(weights_path)

    return model
