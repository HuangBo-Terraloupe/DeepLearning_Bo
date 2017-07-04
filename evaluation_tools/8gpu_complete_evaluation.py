import cv2
import yaml
import numpy as np
import pandas as pd
from scipy import misc
from keras.models import model_from_json
from keras.backend import set_image_dim_ordering
from sklearn.metrics import confusion_matrix
from image_tools import  compress_as_label, normalize_image_channelwise
from tools.load_and_transfer8GPU import load_and_transfer


def predict_single(model, img_patch):
    """
    Predicts the response maps of a single image map. Model should output the response map of same
    size as img_patch.
    :param model: A keras model.
    :param img_patch: Image patch of form (batch_size, h, w, channels).
    :return:
    """
    patch_size = model.input_shape[1:3]
    #img_patch_norm = normalize_image_channelwise(img_patch)
    img_patch_norm = np.expand_dims(img_patch, 0)
    response_map = model.predict(img_patch_norm)
    return np.reshape(response_map.squeeze(), patch_size + (response_map.shape[-1],))

def predict_complete(model, img_data):
    """
    Predicts response map of image having size greater than model input. The response map for the
     image is calculated by finding response maps for smaller overlapping images.
    :param model: Model on which to run the predictions.
    :param img_data: Image of shape (h, w, n_channels)
    :return: Predicted response map
    """
    patch_size = model.input_shape[1:3]
    step_size = (0.8 * np.array(patch_size)).astype(np.uint32)
    # resolution of response map should be same as img_data and channels should be no of channel in model output.
    out_response = np.zeros(img_data.shape[:2] + (model.output_shape[-1],))
    for i in range(0, img_data.shape[0], step_size[0]):
        x = img_data.shape[0] - patch_size[0] if i + patch_size[0] > img_data.shape[0] else i
        for j in range(0, img_data.shape[1], step_size[1]):
            y = img_data.shape[1] - patch_size[1] if j + patch_size[1] > img_data.shape[1] else j
            response_map = predict_single(model, img_data[x:x + patch_size[0], y:y + patch_size[1], :])
            out_response[x:x + patch_size[0], y:y + patch_size[1], :] = np.maximum(
                out_response[x:x + patch_size[0], y:y + patch_size[1], :], response_map)
    return out_response



class Complete_evl:

    def __init__(self, model, nb_classes, mean=None):
        self.model = model
        self.nb_classes = nb_classes
        self.mean = mean


    def evluation_single(self, image, label):

        if self.mean == None:
            normal_image = normalize_image_channelwise(image)

        else :
            normal_image = image - self.mean
            normal_image = normal_image / 255.

        prediction = predict_complete(self.model, normal_image)
        prediction = compress_as_label(prediction)

        label = label.flatten()
        prediction = prediction.flatten()

        cm = confusion_matrix(y_true=label, y_pred=prediction, labels=range(self.nb_classes))

        return cm

    def complete_evaluation(self, dataset_file, labels_index):

        spec = yaml.load(open(dataset_file, "rb").read())
        file_names = spec["validation"]["images"]
        masks_names = spec["validation"]["labels"]

        nb_samples = len(file_names)

        print "the number of evaluation samples are:", nb_samples

        cm = np.zeros(shape=(self.nb_classes, self.nb_classes))
        for i in range(nb_samples):
            print "-----deal with the" + " "+ str(i) + " " + 'image-------'

            image2evl = cv2.imread(file_names[i], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            label = cv2.imread(masks_names[i], -1)
            cm_new = self.evluation_single(image2evl, label)
            cm = cm + cm_new


        # total accuracy
        tp = np.trace(cm)
        tp = float(tp)
        accuracy = tp / np.sum(cm)

        # recall and precision
        recall = np.zeros(self.nb_classes, dtype=float)
        precision = np.zeros(self.nb_classes, dtype=float)
        f1 = np.zeros(self.nb_classes, dtype=float)
        nb_samples = np.zeros(self.nb_classes)

        for i in range(self.nb_classes):
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

        for i in range(self.nb_classes):
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

        return cm, accuracy, pd.DataFrame(dic_report, index=labels_index)

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
              'trees (high vegetation)', 'water', 'fallow land', 'sand / rock']

    # create the object
    unet_evl = Complete_evl(model=model, nb_classes=11, mean=[114.96066323, 116.50405346, 102.74354111])
    cm, total_acc, report = unet_evl.complete_evaluation(dataset_file, index)
    print "The confusion matrix:", cm
    print "The total acc is:", total_acc
    print "The report:", report