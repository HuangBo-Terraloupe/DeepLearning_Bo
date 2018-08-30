import os
import cv2
import numpy as np

from time import time
from glob import glob

from keras.models import Model
from keras.utils import np_utils
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam

def normalization(x, mean):
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x

def build_vgg_model(image_width, image_hight, nb_classes):

    img_input = Input(shape=(image_width, image_hight, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # training model
    base_model = Model(img_input, x, name='base_model')
    base_model.load_weights('/home/bo/Desktop/pre_trained_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print('loading vgg pre-trained weights ...')

    x = base_model.layers[-1].output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    prediction = Dense(nb_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=img_input, outputs=prediction)
    print(model.summary())
    return model

def build_self_made_model(image_width, image_hight, nb_classes):

    img_input = Input(shape=(image_width, image_hight, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    prediction = Dense(nb_classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, prediction, name='linjiali')
    print(model.summary())
    return model

def medicine_classifier_training_run(train_folder, test_folder, image_width, image_hight, model_save_folder):
    '''Training a classifier for medicine classification
    Args:
        train_folder:
        test_folder:
        image_width:
        image_hight:
        model_save_folder:

    Returns:

    '''

    # class_mapping
    class_mapping = {}
    for i, category in enumerate(os.listdir(train_folder)):
        class_mapping[category] = i
    print('The class mapping is', class_mapping)
    print('The number of total classes are:', len(class_mapping))

    # training and testing data statistics
    num_train_samples = 0
    for img_folder in os.listdir(train_folder):
        num_train_samples = num_train_samples + len(glob(os.path.join(train_folder, img_folder) + "/*.jpg"))
        num_train_samples = num_train_samples + len(glob(os.path.join(train_folder, img_folder) + "/*.JPG"))
    print('total number of training samples are:', num_train_samples)

    num_test_samples = 0
    for img_folder in os.listdir(test_folder):
        num_test_samples = num_test_samples + len(glob(os.path.join(test_folder, img_folder) + "/*.jpg"))
        num_test_samples = num_test_samples + len(glob(os.path.join(test_folder, img_folder) + "/*.JPG"))
    print('total number of testing samples are:', num_test_samples)

    # load training data
    train_images = np.zeros([num_train_samples, image_width, image_hight, 3])
    train_labels = np.zeros(num_train_samples, dtype=int)
    train_data_index = 0

    for category in os.listdir(train_folder):
        img_folder_path = os.path.join(train_folder, category)
        image_paths = os.listdir(img_folder_path)
        for image_path in image_paths:
            image = cv2.imread(os.path.join(img_folder_path, image_path))
            image = cv2.resize(image, (image_hight, image_width))
            train_images[train_data_index] = image
            train_labels[train_data_index] = int(class_mapping[category])
            train_data_index = train_data_index + 1

    # load testing data
    test_images = np.zeros([num_test_samples, image_width, image_hight, 3])
    test_labels = np.zeros(num_test_samples, dtype=int)
    test_data_index = 0

    for category in os.listdir(test_folder):
        img_folder_path = os.path.join(test_folder, category)
        image_paths = os.listdir(img_folder_path)
        for image_path in image_paths:
            image = cv2.imread(os.path.join(img_folder_path, image_path))
            image = cv2.resize(image, (image_hight, image_width))
            test_images[test_data_index] = image
            test_labels[test_data_index] = int(class_mapping[category])
            test_data_index = test_data_index + 1

    # training and test data pre-processing
    train_labels = np_utils.to_categorical(train_labels, num_classes=len(class_mapping))
    test_labels = np_utils.to_categorical(test_labels, num_classes=len(class_mapping))

    # calculating the mean
    BGR_mean = [train_images[..., 0].mean(), train_images[..., 1].mean(), train_images[..., 2].mean()]
    print('The BGR mean of image dataset are:', BGR_mean)

    train_images = normalization(train_images, BGR_mean)
    test_images = normalization(test_images, BGR_mean)

    # bulding model
    model = build_vgg_model(image_width, image_hight, nb_classes=len(class_mapping))
    #model = build_self_made_model(image_width, image_hight, nb_classes=len(class_mapping))

    # optimizer
    adam = Adam(lr=1e-6)

    # compile the model
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # callbacks
    filepath = os.path.join(model_save_folder, 'model.h5py')
    weights = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                              mode='auto', period=1)
    tensorboard = TensorBoard(log_dir=os.path.join(model_save_folder, "logs").format(time()))

    print('start training ...')
    # Another way to train the model
    model.fit(train_images, train_labels, epochs=50, batch_size=5, callbacks=[weights, tensorboard], validation_data=(test_images, test_labels))



if __name__ == '__main__':
    train_folder = '/home/bo/Desktop/DataSet/Train'
    test_folder = '/home/bo/Desktop/DataSet/Test'
    image_width = 224
    image_hight = 224
    model_save_folder = '/home/bo/Desktop/DataSet/output'
    medicine_classifier_training_run(train_folder, test_folder, image_width, image_hight, model_save_folder)

