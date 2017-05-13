import os
import random

import numpy as np
import rasterio
import yaml

from dataset import Dataset
import logger

from utils.collection import zip_list_generator
from utils.image.image_io import (get_label_mask,
                                                  load_image_as_blocks,
                                                  load_image_as_window,
                                                  load_labels_as_blocks,
                                                  load_labels_as_window)

from utils.image.image_utils import (divide_in_blocks,
                                                     sample_window,
                                                     normalize_image_channelwise
                                                     )
from utils.ml.dim_ordering import dim_correction_decorator

from utils.meta_schema import MetaSchema
from definitions import image_dataset_schema

np.random.seed(42)


class ImageDataset(Dataset):
    """load dataset.

    Args:
        filename (string): path to description file of the dataset
        image_blocksize (tuple): blocksize (w,h)
        label_blocksize (tuple): blocksize (w,h)
        channels (array): channels to use as an array of indices
                          (rgb => [0,1,2])
                          None or 'all': use all available channels
        normalize (bool): perform mean centralization of training data
        flip_channels (bool): flip channel order (ie. load rgb as bgr)
        binary_label (string): use this class for binary classification
                            if None, use all classes
        samples_per_epoch (int): fixed number of samples per epoch
        select_positive_samples (bool): select patches, that contain at least one pixel of foreground
        positive_negative_ratio (float): ratio of foreground to background samples (patchwise)
        include_validation_for_training (bool): use also the validation data for training. this is useful,
                                                if validation/test split is performed externaly
    """

    # schema for the consumed config file
    schema = MetaSchema(image_dataset_schema)

    def __init__(self,
                 filename,
                 image_blocksize=(32, 32),
                 label_blocksize=(32, 32),
                 use_channels=None,
                 normalize=True,
                 normalization_mean=None,
                 flip_channels=False,
                 binary_label=None,
                 samples_per_epoch=None,
                 select_positive_samples=False,
                 positive_negative_ratio=0.25,
                 include_validation_for_training=False):

        self.image_blocksize = image_blocksize
        self.label_blocksize = label_blocksize
        self._samples_per_epoch = samples_per_epoch
        self.normalize = normalize
        # Redundant as channels can be flipped using use_channels
        self.flip_channels = flip_channels
        self.select_positive_samples = select_positive_samples
        self.positive_negative_ratio = positive_negative_ratio

        with open(filename) as fp:
            spec = yaml.load(fp.read())

            self.schema.validate(spec)

            if include_validation_for_training:
                logger.info("Merge %d validation images/labels into training set" %
                            len(spec["validation"]["images"]))
                spec["training"]["images"] += spec["validation"]["images"]
                spec["training"]["labels"] += spec["validation"]["labels"]
            self.spec = spec

        Dataset.__init__(self, filename,
                         use_channels=use_channels,
                         binary_label=binary_label,
                         samples_per_epoch=samples_per_epoch)

        self.global_training_mean = None

        if flip_channels:
            if self.n_channels > 3:
                logger.warn(
                    "Flipping channels with n_channels > 3. You probably only want to flip 'rgb' images")
            else:
                logger.info("Flip channels 'rgb' to 'bgr'")

        if normalize:
            if normalization_mean is not None:
                self.global_training_mean = normalization_mean
                logger.info("Input mean for dataset %s" %
                            self.global_training_mean)
            else:
                logger.info("Compute global normalization mean")
                self.global_training_mean = self._calculate_global_training_mean()
                logger.info("Computed mean for dataset %s" %
                            self.global_training_mean)

        if select_positive_samples:
            logger.info("Analyze sample distribution")

            self.sample_masks = self._build_sample_masks()
            n_positive = (self.sample_masks == True).sum()
            n_negative = (self.sample_masks == False).sum()

            self.n_negative_samples = min(
                n_positive * 1.0 / self.positive_negative_ratio, n_negative)
            self._samples_per_epoch = self.n_negative_samples + n_positive

            logger.info("Number of positive / negative samples: %d / %d = %.3f" %
                        (n_positive, n_negative, n_positive / float(n_negative)))
            if n_positive / self.positive_negative_ratio >= n_negative:
                logger.info(
                    "Sufficient ratio of positive / negative. Use alle samples.")
                self.select_positive_samples = False
            else:
                logger.info("Target positive / negative ratio = %.2f" %
                            self.positive_negative_ratio)
                logger.info("Select %d random negative samples" %
                            self.n_negative_samples)

    def calculate_class_weights(self):
        """ Calculates class specific weights from the training set. The class
        weight is calculated as its inverted relative frequency in the training
        set."""
        if hasattr(self, "_Y_training"):
            n_classes = self._Y_training.shape[-1]
            y = self._Y_training.argmax(axis=2)
            frequencies = np.histogram(y, bins=n_classes)[0]
        else:
            label_files = self.spec["training"]["labels"]
            frequencies = None
            for label_file in label_files:
                labels = self._load_labels(label_file)
                counts = (labels > 0).sum(axis=(0, 1))
                if frequencies is not None:
                    frequencies += counts
                else:
                    frequencies = counts
        class_freq = frequencies.astype(np.float64) / frequencies.sum()
        class_weights = 1 / class_freq
        return dict(zip(range(class_weights.shape[0]), class_weights.tolist()))

    @dim_correction_decorator
    def batch_data_generator(self, batch_size, subset, sampling_method='shuffle_sequential'):
        if self.select_positive_samples:
            logger.info(
                "Option 'select_positive samples' ignores sampling method")
            return self.select_positive_batch_generator(batch_size, subset)

        logger.info("Use sampling_method=%s for subset=%s" %
                    (sampling_method, subset))
        if sampling_method == 'shuffle_random':  # shuffle with random access
            return self.shuffled_batch_generator(subset, batch_size)
        elif sampling_method == 'shuffle_sequential':  # shuffle with sequential access
            return self.shuffle_sequential_batch_generator(subset, batch_size)
        elif sampling_method == 'random':
            return self.random_batch_generator(subset, batch_size)
        elif sampling_method == 'sequential':
            return self.sequential_batch_generator(subset, batch_size)
        else:
            raise Exception("Sampling method '%s' not supported" %
                            sampling_method)

    def load_samples(self, subset):
        """Filter patches by foreground / background for sparse foreground datasets.

        Uses samples that contain foreground images, plus a given ratio of negative samples.
        """

        image_files = [f for f in self.spec[subset]["images"]]
        label_files = [f for f in self.spec[subset]["labels"]]

        logger.info("Loading % samples from %d files" %
                    (subset, len(image_files)))

        # NOTE assuming all images have the same shape!
        img_shape = self._read_image_shape(image_files[0])
        block_size = self.image_blocksize

        blocks = divide_in_blocks(img_shape, block_size)

        positive_indices = zip(*np.where(self.sample_masks == True))
        negative_indices = zip(*np.where(self.sample_masks == False))
        random.shuffle(negative_indices)
        random.shuffle(positive_indices)

        image_blocks = []
        label_blocks = []
        sample_indices = positive_indices + \
            negative_indices[:int(self.n_negative_samples)]
        n_samples = len(sample_indices)

        for file_index, block_index in sample_indices:
            block = blocks[block_index]
            label_blocks.append(self._load_labels_as_window(
                label_files[file_index], block))
            image_blocks.append(self._load_image_as_window(
                image_files[file_index], block))

        logger.info("%d sample loaded" % n_samples)

        X = np.array(image_blocks)
        Y = np.array(label_blocks)
        self._X_training = X
        self._Y_training = Y
        return X, Y

    def select_positive_batch_generator(self, batch_size, subset):
        X, Y = self.load_samples(subset)
        return zip_list_generator(X, Y, batch_size)

    def random_batch_generator(self, subset, batch_size=40):
        """Generating batches of completely random samples infinitely.

        Args:
            batch_size:
            subset: can be training, testing, or validation
        Return:
            (generator): generate a batch of random samples (image, label)
        """

        while True:
            samples = [self._random_sample(subset=subset)
                       for i in range(batch_size)]

            batch_x = np.array([x for (x, _) in samples])
            batch_y = np.array([y for (_, y) in samples])

            yield (batch_x, batch_y)

    def shuffled_batch_generator(self, subset, batch_size=40):
        """Lazy loading of training batch from training samples. Covers the whole
        training set, but in random order.

        Args:
            batch_size:
            subset: can be training, testing or validation
        Return:
            (generator): generate a batch of random samples (image, label)
        """
        image_files = self.spec[subset]["images"]
        label_files = self.spec[subset]["labels"]
        training_size = len(image_files)

        # NOTE assuming all images have the same shape!
        img_shape = self._read_image_shape(image_files[0])

        random_indices = range(training_size)
        random.shuffle(random_indices)

        block_size = self.image_blocksize
        random_subdivisions = []

        for i in range(training_size):
            blocks = divide_in_blocks(img_shape, block_size)
            random.shuffle(blocks)
            random_subdivisions.append((i, blocks))

        training_samples = []
        while True:
            for i, blocks in random_subdivisions:
                image_file = image_files[i]
                label_file = label_files[i]

                for block in blocks:
                    image_block = self._load_image_as_window(image_file, block)
                    label_block = self._load_labels_as_window(
                        label_file, block)
                    training_samples.append((image_block, label_block))
                    if len(training_samples) == batch_size:
                        batch_x = np.array([x for (x, _) in training_samples])
                        batch_y = np.array([y for (_, y) in training_samples])
                        yield (batch_x, batch_y)

                        training_samples = []

    def shuffle_sequential_batch_generator(self, subset, batch_size=40):
        """Lazy loading of training batch from training samples. Covers the whole
        training set, but in random order.

        Args:
            batch_size:
            subset: can be training, testing or validation
        Return:
            (generator): generate a batch of random samples (image, label)
        """
        image_files = self.spec[subset]["images"]
        label_files = self.spec[subset]["labels"]
        training_size = len(image_files)
        random_indices = range(training_size)
        random.shuffle(random_indices)

        training_samples = []
        random_block_indices = None
        while True:
            for i in random_indices:
                image_file = image_files[i]
                label_file = label_files[i]

                image_blocks = self._load_image(image_file)
                label_blocks = self._load_labels(label_file)

                if not random_block_indices:
                    random_block_indices = range(len(image_blocks))
                    random.shuffle(random_block_indices)

                for j in random_block_indices:
                    training_samples.append((image_blocks[j], label_blocks[j]))
                    if len(training_samples) == batch_size:
                        batch_x = np.array([x for (x, _) in training_samples])
                        batch_y = np.array([y for (_, y) in training_samples])
                        yield (batch_x, batch_y)

                        training_samples = []

    def sequential_batch_generator(self, subset, batch_size=40):
        """Lazy loading of training data files.

        Args:
            batch_size (integer): maximum number of patches per batch.
            subset: can be training, testing or validation
        """
        image_files = self.spec[subset]["images"]
        label_files = self.spec[subset]["labels"]

        while 1:
            for image_file, label_file in zip(image_files, label_files):
                x = self._load_image(image_file)
                y = self._load_labels(label_file)

                for i in range(0, x.shape[0], batch_size):
                    batch_size = min(batch_size, x.shape[0] - i)
                    yield x[i:i + batch_size, ...], y[i:i + batch_size, ...]
                    # FIXME: Better fill batches by loading first samples from next
                    # image when reaching last batch.

    def _calculate_global_training_mean(self):
        filenames = self._absolute_filenames(subset="training", part="images")
        with rasterio.open(filenames[0]) as src:
            n_channels = src.count

        mean = np.zeros(n_channels)
        n_images = len(filenames)
        for filename in filenames:
            with rasterio.open(filename) as src:
                image = src.read()
                mean += image.mean(axis=(1, 2)) / n_images

        if self.use_channels:
            mean = mean[self.use_channels]

        return mean

    def _absolute_filenames(self, subset=None, part=None):

        files = [f for f in self.spec[subset][part]]
        return [os.path.join(self.prefix, filename) for filename in files]

    def _random_sample(self, subset):
        """Sampling of random training data

        Args:
            subset: can be training, testing or validation
        Return:
            (tuple): random sample (image, label) for the given blocksize
        """

        label_files = self.spec[subset]["labels"]
        image_files = self.spec[subset]["images"]
        blocksize = self.image_blocksize

        assert len(image_files) > 0
        assert len(image_files) == len(label_files)

        rand_index = random.randint(0, len(image_files) - 1)
        img_file, label_file = image_files[rand_index], label_files[rand_index]
        img_shape = self._read_image_shape(img_file)

        window = sample_window(img_shape, blocksize)
        image_block = normalize_image_channelwise(self._load_image_as_window(img_file, window))
        label_block = self._load_labels_as_window(label_file, window)

        return (image_block, label_block)

    def _load_subset(self, subset):
        label_files = self.spec[subset]["labels"]
        image_files = self.spec[subset]["images"]

        x = np.concatenate([self._load_image(f) for f in image_files])
        y = np.concatenate([self._load_labels(f) for f in label_files])

        return x, y

    def _read_image_shape(self, filename):
        absolute_path = os.path.join(self.prefix, filename)
        with rasterio.open(absolute_path) as src:
            return src.shape

    def _load_image_as_window(self, filename, window):
        absolute_path = os.path.join(self.prefix, filename)

        return load_image_as_window(absolute_path,
                                    normalize=self.normalize,
                                    use_channels=self.use_channels,
                                    window=window,
                                    normalization_mean=self.global_training_mean)

    def _load_labels_as_window(self, filename, window):

        absolute_path = os.path.join(self.prefix, filename)
        label_image = load_labels_as_window(absolute_path,
                                            n_labels=self.n_labels_gt,
                                            window=window)

        if self.binary_label is not None:
            label_image = get_label_mask(label_image,
                                         self._get_label_index(self.binary_label))

        return label_image

    def _load_image(self, filename):
        absolute_path = os.path.join(self.prefix, filename)

        return load_image_as_blocks(absolute_path,
                                    normalize=self.normalize,
                                    blocksize=self.image_blocksize,
                                    use_channels=self.use_channels,
                                    normalization_mean=self.global_training_mean)

    def _load_labels(self, filename):

        absolute_path = os.path.join(self.prefix, filename)

        labels = load_labels_as_blocks(absolute_path,
                                       n_labels=self.n_labels_gt,
                                       blocksize=self.label_blocksize)

        if self.binary_label is not None:
            labels = get_label_mask(labels,
                                    self._get_label_index(self.binary_label))

        return labels

    # TODO memoize
    @property
    def samples_per_epoch(self):
        if self._samples_per_epoch:
            return min(self._samples_per_epoch, self.n_training_samples)
        else:
            return self.n_training_samples

    @property
    def n_training_samples(self):
        return self.n_training_files * self.n_samples_per_file

    @property
    def n_validation_samples(self):
        return self.n_validation_files * self.n_samples_per_file

    @property
    def n_samples_per_file(self):
        # TODO use memoized-property here and everywhere else
        cache_key = "_n_samples_per_file"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)
        else:
            image = self._load_image(self.spec["training"]["images"][0])
            setattr(self, cache_key, image.shape[0])
            return image.shape[0]

    @property
    def n_training_files(self):
        return len(self.spec["training"]["images"])

    @property
    def n_validation_files(self):
        return len(self.spec["validation"]["images"])

    def _build_sample_masks(self):
        masks = []
        for label_file in self.spec['training']['labels']:
            labels = self._load_labels(label_file)
            mask = labels[:, :, self.background_index].sum(
                axis=-1) < labels.shape[1]
            masks.append(mask)
        return np.array(masks)

    def __getitem__(self, key):
        return self.spec[key]

    @dim_correction_decorator
    def complete_dataset_generator(self, batch_size, subset):
        """
        Generator to iterate complete dataset once. Useful for evaluation.
        :param batch_size: Batch size.
        :param subset: test, validation or
        :return: Image data and labels.
        """
        image_files = self.spec[subset]['images']
        label_files = self.spec[subset]['labels']
        images_batch = []
        labels_batch = []
        image_shape = self._read_image_shape(image_files[0])
        window = ((0, image_shape[0]), (0, image_shape[1]))
        for i in range(len(image_files)):
            label_file = label_files[i]
            image_file = image_files[i]
            image_data = load_image_as_window(image_file, normalize=False, window=window,
                                              use_channels=self.use_channels)
            label_data = load_labels_as_window(
                label_file, n_labels=4, window=window)
            images_batch.append(image_data)
            labels_batch.append(label_data)
            if (i + 1) % batch_size == 0:
                yield images_batch, labels_batch
                images_batch = []
                labels_batch = []