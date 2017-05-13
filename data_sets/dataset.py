import numpy as np
import re
import os

from utils import logger

np.random.seed(42)


class Dataset:
    """load dataset.

    Args:
        filename (string): path to description file of the dataset
        channels (array): channels to use as an array of indices
                          (rgb => [0,1,2])
                          None or 'all': use all available channels
        binary_label (string): use this class for binary classification
                            if None, use all classes
        samples_per_epoch (int): Fixed number of samples per epoch
    """

    def __init__(self,
                 filename,
                 use_channels=None,
                 binary_label=None,
                 samples_per_epoch=None):

        # set to None, which represent all available channels
        if use_channels == "all":
            use_channels = None

        self.use_channels = use_channels
        self.binary_label = binary_label
        self._samples_per_epoch = samples_per_epoch
        self.dataset_file = filename

    # TODO use memoization decorator
    @property
    def validation_set(self):
        subset_name = "validation"
        cache_key = "_cache_%s" % subset_name
        if not hasattr(self, cache_key):
            setattr(self, cache_key, self._load_subset(subset_name))
        return getattr(self, cache_key)

    @property
    def test_set(self):
        subset_name = "test"
        cache_key = "_cache_%s" % subset_name
        if not hasattr(self, cache_key):
            setattr(self, cache_key, self._load_subset(subset_name))
        return getattr(self, cache_key)

    def random_validation_samples(self, count):

        x, y = self.validation_set
        total_samples = x.shape[0]
        idxs = np.random.choice(total_samples, count)

        x = x[idxs]
        y = y[idxs]

        return x, y

    def _get_label_index(self, label_name):
        cls_item = filter(lambda cls: cls["name"] == label_name,
                          self.classes)[0]
        return cls_item["index"]

    @property
    def name(self):
        return self.spec["name"]

    @property
    def prefix(self):
        if 'prefix' in self.spec:
            return self.spec["prefix"]

        return os.path.dirname(self.dataset_file)

    @property
    def type(self):
        return self.spec["ground_truth"]["type"]

    @property
    def classes(self):
        return self.spec["ground_truth"]["classes"]

    @property
    def n_labels(self):
        """number of labels in training samples"""
        if self.binary_label is not None:
            return 1
        return self.n_labels_gt

    @property
    def n_labels_gt(self):
        """number of labels in ground truth"""
        return len(self.classes)

    @property
    def spec(self):
        return self.spec

    @property
    def image_type(self):
        return self.spec["data"]["type"]

    @property
    def image_channels(self):
        return self.spec["data"]["channels"]

    @property
    def n_channels(self):
        if self.use_channels:
            return len(self.use_channels)
        else:
            return len(self.image_channels)

    @property
    def samples_per_epoch(self):
        raise NotImplementedError("")

    @property
    def background_index(self):
        background_index = self.spec['ground_truth'].get('background_index')
        if background_index is not None:
            return background_index
        else:
            possible_indices = [c['index']
                                for c in self.classes
                                if re.match(r'.*background.*', c['name'])]
            if len(possible_indices) > 0:
                background_index = possible_indices[0]
            else:
                background_index = 0

            logger.info("Background index not specified. Guessing: '%s'" %
                        background_index)
            self.background_index = background_index
            return background_index

    @property
    def class_names(self):
        return [c['name'] for c in self.classes]
