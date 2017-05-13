import random

import numpy as np
import skimage
from skimage.util import view_as_windows


def normalize_block_image(image):
    """
    Normalize a block-view image.

    featurewise centering and std_normalization
    Reshaping has to be performed to allow broadcasting

    Args:
        image (nparray): image with shape (nblocks, nchannels, w, h)
    Return:
        normalized image (nparray)
    """
    n_channels = image.shape[-1]
    image -= image.mean(axis=(0, 1, 2)).reshape((1, 1, n_channels))
    image /= image.std(axis=(0, 1, 2)).reshape((1, 1, n_channels))
    return image


def flip_axis(image, axis=0):
    slice_index = [slice(None)] * len(image.shape)
    slice_index[axis] = slice(None, None, -1)
    return image[slice_index]


def normalize_image(image, normalization_mean):
    """Normalize a multi channel images

    Args:
        image (ndarray): image with shape (nchannels, w, h)
        mean (ndarray): mean per channel
    Return:
        normalized image (ndarray)
    """
    n_channels = image.shape[-1]
    assert len(normalization_mean) == n_channels, "Length of normalization_mean must match the image channels"
    assert image.min() >= 0.0, "Image has negative values, might be already normalized!"
    if image.max() > 240.0 and normalization_mean.max() < 1.0:
        print "Warning! Image is not normalized to [0, 1]! Dividing by 255.."
        image /= 255.0

    normalization_mean = np.asarray(normalization_mean, dtype=image.dtype)
    image -= normalization_mean.reshape((1, 1, n_channels))
    return image


def crop_blocksize(image, blocksize):
    """ Crops the given image image [channels, rows, cols] such that its
    dimensions are a multiple of the blocksize tuple, allowing for evenly
    splitting the image into block using `view_as_blocks`.
    """
    h, w = image.shape[:2]  # cut out channel
    target_rows = (h // blocksize[0]) * blocksize[0]
    target_cols = (w // blocksize[1]) * blocksize[1]

    row_start = (h - target_rows) // 2
    col_start = (w - target_cols) // 2

    row_end = row_start + target_rows
    col_end = col_start + target_cols
    if image.ndim == 2:
        return image[row_start:row_end, col_start:col_end]
    elif image.ndim == 3:
        return image[row_start:row_end, col_start:col_end, :]


def crop_overlapping_windows(windows, overlap):
    """
    Crop windows to remove overlap

    Arguments:

        windows: tensor of overlapping windows (n_windows, n_windows, block_size, block_size,
                                                n_channels))


    """
    blocksize = windows.shape[-3:-1]
    border = overlap / 2
    ystart, yend = border, blocksize[0] - border
    xstart, xend = border, blocksize[1] - border
    return windows[:, :, ystart:yend, xstart:xend, :]


def decompress_labels(label_image, n_labels):
    """Decompress semseg label image.

    Decompose uint8 label image with enumerated class indices into
    float32 tensor with one mask binary layer per class.
    Output shape will be: (nclasses, w, h)

    Args:
        label_image (ndarrray): compressed uint 8 label image, with an integer
                                value representing the class
        n_labels (int): total number of classes
    :return (h, w, n_labels) numpy array.
    """

    layers = np.zeros(label_image.shape + (n_labels,), dtype=np.float32)

    for i in range(n_labels):
        layers[:, :, i] = label_image == i

    return layers


def sample_window(shape, blocksize):
    """Sample a random window for of the given blocksize

    Args:
        shape (tuple): shape (h, w) for image size
        blocksize (tuple): blocksize of resulting block
    Return:
        a random within the image boundaries
    """
    height, width = shape
    rand_x = random.randint(0, width - blocksize[0])
    rand_y = random.randint(0, height - blocksize[1])
    rand_window = ((rand_y, rand_y + blocksize[1]), (rand_x, rand_x + blocksize[0]))
    return rand_window


def divide_in_blocks(shape, block_size):
    """Subdivides an area of the given shape into blocks of the
    given blocks size

    Args:
        shape (tuple): shape (h, w) for image size
        blocksize (tuple): blocksize of resulting blocks
    Return:
        list of blocks in the rasterio window format
    """

    h, w = shape
    stride_y, stride_x = block_size
    target_cols = w / stride_x
    target_rows = h / stride_y

    blocks = []

    for j in xrange(target_rows):
        for i in xrange(target_cols):
            x1 = i * stride_x
            x2 = x1 + stride_x
            y1 = j * stride_y
            y2 = y1 + stride_y

            block = ((y1, y2), (x1, x2))
            blocks.append(block)

    return blocks


def split_image(image_data, out_resolution):
    """
    Splits an image into smaller images of resolution out_resolution.
    :param Image to be splitted.
    :param Resolution of small images.
    :return Grid of splitted images. The size of the grid will be
    image_data.shape/out_resolution.
    """
    sub_image_shape = (image_data.shape[0],) + out_resolution
    step_shape = (1,) + out_resolution
    sub_images = skimage.util.view_as_windows(image_data, sub_image_shape, step_shape)
    sub_images = sub_images.squeeze(axis=0)
    return sub_images


def normalize_image_channelwise(image_data):
    """
    Normalizes each channel of the image_data to 0 zero mean and unit std. All image operation should have
    last dimension as channel except while passing image to/from keras model.
    :param image_data: Input image of shape (...,n_channels)
    :return: Normalized image Image normalized across each channel.
    """
    n_channels = image_data.shape[-1]
    normalized_img = np.zeros(image_data.shape)
    for i in range(n_channels):
        channel_data = image_data[..., i]
        normalized_img[..., i] = (channel_data - channel_data.mean()) / (channel_data.std() + 1e-9)
    return normalized_img


def compress_as_label(response_maps, threshold=0.5):
    """
    Compresses probability  response maps as labels. First channel must be background response map.
    :param response_maps: List of response maps of shape (...,_channels)
    :return: Compressed label of shape (...) with each location containing the index of channel with max prob.
    """
    n_channels = response_maps.shape[-1]
    if n_channels == 1:  # binary segmentation
        out_response_map = np.zeros(response_maps.shape)
        out_response_map[response_maps > threshold] = 1
        return np.squeeze(out_response_map, axis=-1)
    else:
        return np.argmax(response_maps, -1)


def image_as_windows(image, blocksize=(64, 64), overlap=0, flatten=True):
    """
    Returns image as windows of blocksize along all image channels.
    :param image: (h, w, channels) image.
    :param blocksize: Size of window.
    :param overlap: Overlapping between different windows.
    :param flatten: If all windows are to be flattened as vector.
    :return: (n_windows, blocksize[0], blocksize[1], n_channels) If flatten is True
    """
    n_channels = image.shape[-1]
    stepsize = (blocksize[0] - overlap, blocksize[1] - overlap, 1)
    windows = view_as_windows(image, blocksize + (n_channels,), stepsize)
    if flatten:
        windows = windows.reshape((-1,) + windows.shape[3:])
    else:
        # all except 2nd dimension as all channels are taken
        windows = windows.reshape(windows.shape[:2] + windows.shape[3:])
    return windows
