import yaml
import numpy as np
import pandas as pd
from scipy import misc
from keras.models import model_from_json
from keras.backend import set_image_dim_ordering
from sklearn.metrics import confusion_matrix
from image_tools import  compress_as_label, predict_complete,  normalize_image_channelwise



class Complete_evl:

    def __init__(self, model, nb_classes):
        self.model = model
        self.nb_classes = nb_classes


    def evluation_single(self, image, label):

        normal_image = normalize_image_channelwise(image)
        prediction = predict_complete(self.model, normal_image)
        prediction = compress_as_label(prediction)

        label = label.flatten()
        prediction = prediction.flatten()

        cm = confusion_matrix(y_true=prediction, y_pred=label, labels=range(self.nb_classes))

        return cm

    def complete_evaluation(self, dataset_file, labels_index):

        with open(dataset_file) as fp:
            spec = yaml.load(fp.read())

        nb_samples = len(spec["validation"]["images"])

        print "the number of evaluation samples are:", nb_samples

        cm = np.zeros(shape=(self.nb_classes, self.nb_classes))
        for i in range(nb_samples):
            print "-----deal with the" + " "+ str(i) + " " + 'image-------'

            image2evl = misc.imread(spec["validation"]["images"][i])
            labels = misc.imread(spec["validation"]["labels"][i])

            cm_new = self.evluation_single(image2evl, labels)
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
    json_file = open('/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/model_540.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)


    model.load_weights("/home/huangbo/objectdetection/objectdetection/huangbo_ws/models/05.11_unet_540/weights_best.hdf5")

    # load image
    dataset_file = "/home/huangbo/objectdetection/objectdetection/huangbo_ws/nordhorn_2.yml"

    index = ['background', 'building', 'asphalt/concrete', 'railway',
                                              'cars', 'flat vegetation', 'bushes (medium vegetation)',
                                              'trees (high vegetation)', 'water',
                                              'fallow land', 'sand / rock']

    # create the object
    unet_evl = Complete_evl(model=model, nb_classes=11)
    cm, total_acc, report = unet_evl.complete_evaluation(dataset_file,index)
    print "The confusion matrix:", cm
    print "The total acc is:", total_acc
    print "The report:", report


