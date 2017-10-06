import os
import yaml
import numpy as np
import cv2


class Calculate_global_mean:
    def __init__(self, data_file):
        with open(data_file, 'r') as fp:
            self.spec = yaml.load(fp.read())

    def get_file_list(self, subset, part):
        directory = self.spec["prefix"]
        if not part in self.spec[subset]:
            return None

        return [os.path.join(directory, f) for f in self.spec[subset][part]]

    def load_img(self, filename, single_channel=False, target_size=None, dtype=np.float32):

        """ Loads images using opencv library.
            This method does not load geo-information
            The order of channels by default is BGR and not RGB
            cv2.IMREAD_ANYDEPTH is set
    
        Args:
            filename: Path of the image file
            single_channel: (optional) If the image consist of only single channel (E.g. segmentation masks)
            target_size: (optional) resizes the image to given size
            dtype: output datatype. By default float32
    
        Returns: image in the form of numpy array
    
        """

        if single_channel:
            im = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        else:
            im = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

        if target_size:
            im = cv2.resize(im, target_size)

        return np.array(im, dtype=dtype)

    def calculate_global_training_mean(self, filenames):
        """ Calculates the training mean.
    
        Args:
            filenames: A list of image files for which mean is calculated.
    
        Returns: numpy array of mean pixels
    
        """
        src = self.load_img(filenames[0])
        n_channels = src.shape[-1]
        mean = np.zeros(n_channels)
        n_images = len(filenames)

        for filename in filenames:
            image = self.load_img(filename)
            image = np.asarray(image, dtype=np.float32)
            mean += image.mean(axis=(0, 1)) / n_images

        return mean


if __name__ == '__main__':
    yml_file = '/home/huangbo/HuangBo_Projects/harvey/training_data_filter_object/harvey.yml'
    cc = Calculate_global_mean(yml_file)
    img_list_training = cc.get_file_list(subset='training', part='images')
    img_list_validation = cc.get_file_list(subset='validation', part='images')
    img_list_test = cc.get_file_list(subset='validation', part='images')
    img_list = img_list_training + img_list_validation + img_list_test
    mean = cc.calculate_global_training_mean(img_list)
    print mean
    print 'last mean:', [122.52059453, 137.99474664, 142.55964144]





