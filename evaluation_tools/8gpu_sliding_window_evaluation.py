import cv2
import yaml
import numpy as np
import pandas as pd
from scipy import misc
from keras.backend import set_image_dim_ordering
from keras.models import model_from_json, load_model
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from image_tools import get_discrete_sliding_window_boxes, compress_as_label, \
    predict_complete, crop_image
from tools.load_and_transfer8GPU import load_and_transfer


class Sliding_window_evl:
    """ Evaluation the model using sliding window

    Args:
        model: a keras model
        box_size: the size of sliding window box, also equal to the input size of network
    """

    def __init__(self, model, box_size, mean=None):
        self.model = model
        self.box_size = box_size
        self.mean = mean

    def evluation_single(self, image):

        boxes_list = get_discrete_sliding_window_boxes(img_size=(image.shape[0],
                                                                 image.shape[1]),
                                                       bbox_size=(self.box_size, self.box_size),
                                                       stride=(self.box_size / 2, self.box_size / 2)
                                                       )

        nb_windows = len(boxes_list)
        nb_row = 0

        for i in range(nb_windows):
            if boxes_list[i][0] == 0:
                nb_row += 1
        nb_col = nb_windows / nb_row

        offset = 0
        for j in range(nb_col):

            for i in range(nb_row):
                img = crop_image(image=image, crop_size=(self.box_size, self.box_size),
                                 offset=(boxes_list[i + offset][0], boxes_list[i + offset][1]))

                img_norm = img - self.mean
                img_norm = img_norm / 255.

                img_norm = np.expand_dims(img_norm, axis=0)
                prediction = self.model.predict(img_norm)
                prediction = compress_as_label(prediction)
                prediction = np.reshape(prediction, (self.box_size, self.box_size))
                crop_prediction = crop_image(image=prediction, crop_size=(self.box_size / 2, self.box_size / 2),
                                             offset=(self.box_size / 4, self.box_size / 4))
                crop_prediction = np.squeeze(crop_prediction, axis=2)

                if i == 0:
                    row_image = crop_prediction
                else:
                    row_image = np.concatenate((row_image, crop_prediction), axis=1)
            offset += nb_row
            if j == 0:
                whole_image = row_image
            else:
                whole_image = np.concatenate((whole_image, row_image), axis=0)

        return whole_image

    def evluation_whole(self, dataset_file, nb_classes, labels_index):

        with open(dataset_file) as fp:
            spec = yaml.load(fp.read())

        spec = yaml.load(open(dataset_file, "rb").read())
        file_names = spec["validation"]["images"]
        masks_names = spec["validation"]["labels"]

        nb_samples = len(spec["validation"]["images"])
        cm = np.zeros(shape=(nb_classes, nb_classes))

        print "The number of evaluation sampels is:", nb_samples

        for i in range(nb_samples):
            print "-----deal with the" + " "+ str(i) + " " + 'image-------'
            image2evl = cv2.imread(file_names[i], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            labels = cv2.imread(masks_names[i], -1)

            # build the sliding window object
            prediction = self.evluation_single(image2evl)
            crop_shape = (prediction.shape[0], prediction.shape[1])
            labels = crop_image(image=labels, crop_size=crop_shape, offset=(self.box_size / 4, self.box_size / 4))

            # confusion matrix
            labels = labels.flatten()
            prediction = prediction.flatten()

            mat_new = confusion_matrix(y_true=prediction, y_pred=labels, labels=range(nb_classes))
            cm = cm + mat_new

        #print "The confusion matrix is:", cm

        # total precision
        tp = np.trace(cm)
        tp = float(tp)
        accuracy = tp / np.sum(cm)
        #print "The total accuracy is:", accuracy

        # recision and recall

        recall = np.zeros(nb_classes, dtype=float)
        precision = np.zeros(nb_classes, dtype=float)
        f1 = np.zeros(nb_classes, dtype=float)
        nb_samples = np.zeros(nb_classes)

        for i in range(nb_classes):
            fp = np.sum(cm[:, i]) - cm[i, i]
            fn = np.sum(cm[i, :]) - cm[i, i]

            if cm[i, i] + fp == 0:
                recall[i] = 0
            else:
                recall[i] = cm[i, i] / (cm[i, i] + fp)

            if cm[i, i] + fn == 0:
                precision[i] = 0
            else:
                precision[i] = cm[i, i] / (cm[i, i] + fn)

        for i in range(nb_classes):
            nb_samples[i] = np.sum(cm[:, i])
            if precision[i] + recall[i] == 0:
                f1[i] = 0
            else:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

        where_are_NaNs = np.isnan(recall)
        recall[where_are_NaNs] = 0

        where_are_NaNs = np.isnan(precision)
        precision[where_are_NaNs] = 0

        dic_report = {"recall": recall,
                      "precision": precision,
                      "f1_score": f1,
                      "number_samples": nb_samples
                      }

        final_report =  pd.DataFrame(dic_report, index= labels_index)
        return cm, accuracy, final_report


if __name__ == "__main__":
    set_image_dim_ordering('tf')


    #load the model
    json_file = '/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/06.18_resnet_256/model.json'
    weights_file = "/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/06.18_resnet_256/model.hdf5"
    model = load_and_transfer(model_file=json_file, weights_file=weights_file)

    # load image
    dataset_file = "/home/huangbo/HuangBo_Projects/data/nordhorn/dataset_500/nordhorn_home.yml"

    index = ['background', 'building', 'asphalt/concrete', 'railway',
                                              'cars', 'flat vegetation', 'bushes (medium vegetation)',
                                              'trees (high vegetation)', 'water',
                                              'fallow land', 'sand / rock']

    unet_evl = Sliding_window_evl(model=model, box_size=256, mean=[114.96066323, 116.50405346, 102.74354111])
    cm, total_acc, report =unet_evl.evluation_whole(dataset_file=dataset_file, nb_classes=11, labels_index=index)

    print "The confusion matrix:", cm
    print "The total acc is:", total_acc
    print "The report:", report
