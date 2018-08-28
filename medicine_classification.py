import os
import cv2
import numpy as np

from time import time
from glob import glob
from PIL import Image

from keras.models import Model
from keras.utils import np_utils
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.layers import Dense, Flatten
from keras.optimizers import Adam


def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()

def normalization(x):
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68

    return x/127.5


def medicine_classifier_training_run(train_folder, test_folder, image_width, image_hight, model_save_folder):

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
            image = read_image_bgr(os.path.join(img_folder_path, image_path))
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
            image = read_image_bgr(os.path.join(img_folder_path, image_path))
            image = cv2.resize(image, (image_hight, image_width))
            test_images[test_data_index] = image
            test_labels[test_data_index] = int(class_mapping[category])
            test_data_index = test_data_index + 1

    # training and test data pre-processing
    train_labels = np_utils.to_categorical(train_labels, num_classes=len(class_mapping))
    test_labels = np_utils.to_categorical(test_labels, num_classes=len(class_mapping))

    train_images = normalization(train_images)
    test_images = normalization(test_images)

    # training model
    base_model = VGG16(input_shape=(image_width, image_hight, 3), weights='imagenet', include_top=False)
    x = base_model.layers[-1].output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    prediction = Dense(len(class_mapping), activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=prediction)
    print(model.summary())

    # optimizer
    adam = Adam(lr=1e-4)

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
    model.fit(train_images, train_labels, epochs=5, batch_size=1, callbacks=[weights, tensorboard], validation_data=(test_images, test_labels))



if __name__ == '__main__':
    train_folder = '/home/terraloupe/DataSet/Train'
    test_folder = '/home/terraloupe/DataSet/Test'
    image_width = 449
    image_hight = 449
    model_save_folder = '/home/terraloupe/DataSet/output'
    medicine_classifier_training_run(train_folder, test_folder, image_width, image_hight, model_save_folder)

