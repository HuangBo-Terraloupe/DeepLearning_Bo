import gdal
import numpy as np
from skimage.util import view_as_blocks
from data_sets.utils.image import image_utils

from image_utils import (crop_blocksize, decompress_labels, normalize_image)
from data_sets.utils import rasterio_utils


def load_geo_reference(filename):
    """ Load geo reference, consisting of projection and geo_transform.
    Args:
        filename (string): image filename
    Return:
        (tuple(string, tuple): (projection, geo_transform)
    """

    ds = gdal.Open(filename)
    projection = ds.GetProjection()
    geo_transform = ds.GetGeoTransform()
    return projection, geo_transform


def load_gtiff(filename, dtype=None):
    """Load geo tiff image.

    Args:
        filename (string): image filename
        dtype (np type): convert to numpy type if, present
    Return:
        (ndarray): image
    """
    ds = gdal.Open(filename)
    image = ds.ReadAsArray()
    if dtype:
        image = image.astype(dtype)
    return image


def load_gtiff_with_geo_reference(filename, dtype=None):
    ds = gdal.Open(filename)
    image = ds.ReadAsArray()
    if dtype:
        image = image.astype(dtype)
    return image, (ds.GetProjection(), ds.GetGeoTransform())


def save_gtiff(image, filename, geo_reference=None, spread=1.0):
    """Save image as geo tiff.

    Args:
        image (ndarray): Input image of shape (n_channels, w, h)
        filename (string): filename to save.
        geo_reference (tuple(string, string)): gdal projection and geo_transform
                                of input image (driver_in.GetProjection() and
                                driver_in.GetGeoTransform())
        spread (float): spread factor (e.g. 255 for uint8)
    """
    driver = gdal.GetDriverByName("GTiff")
    image = image * spread

    if image.ndim == 3:
        n_channels, h, w = image.shape
    elif image.ndim == 2:
        h, w = image.shape
        n_channels = 1
    ds_out = driver.Create(filename, w, h, n_channels, gdal.GDT_Float32)
    if geo_reference:
        projection, geo_transform = geo_reference
        ds_out.SetProjection(projection)
        ds_out.SetGeoTransform(geo_transform)

    # write channels separately

    for i in xrange(n_channels):
        band = ds_out.GetRasterBand(i + 1)  # 1-based index
        if n_channels > 1:
            channel = image[i]
        else:
            channel = image
        band.WriteArray(channel)
        band.FlushCache()

    del ds_out


def load_image_as_window(filename,
                         use_channels=None,
                         normalize=True,
                         window=((0, 32), (0, 32)),
                         normalization_mean=None):
    """Load a single image window.

    Args:
        filename (string): Filename, expects float32, normalized to [0,1]
        channels (array): channels to use as an array of indices (rgb => [0,1,2])
                          if None: use all channels
        normalize (boolean): perform std mean normalization
        window (window): rasterio window ((row_start, row_stop), (col_start, col_stop))
        normalization_mean (ndarray): mean per channel (required if normalize=true)
    Return:
        np-array of the image
    """
    image = rasterio_utils.load_image_data(filename, window=window)


    if use_channels:
        image = image[:, :, use_channels]  # slice channels

    if normalize:
        assert normalization_mean is not None, "No normalization mean given"
        image = normalize_image(image, normalization_mean)

    return image


def load_labels_as_window(filename, n_labels, window=((0, 32), (0, 32))):
    """Load a single label image window.

    Args:
        filename (string): Filename, expects float32, normalized to [0,1]
        n_labels (integer): Number of labels
        window (window): rasterio window ((row_start, row_stop), (col_start, col_stop))
    Return:
        np-array of the image
    """

    label_image = rasterio_utils.load_image_data(filename, window)
    # Label should always has single channel.
    label_image = np.squeeze(label_image, axis=-1)
    label_image = decompress_labels(label_image, n_labels)

    h, w = label_image.shape[:2]
    label_image = label_image.reshape((w * h, n_labels))

    return label_image


def load_image_as_blocks(filename,
                         use_channels=None,
                         normalize=True,
                         blocksize=(64, 64),
                         normalization_mean=None,
                         overlap=0,
                         flatten=True):
    """Load image as blocks of given blocksize.

    Args:
        filename (string): Filename, expects float32, normalized to [0,1]
        channels (array): channels to use as an array of indices (rgb => [0,1,2])
                          if None: use all channels
        normalize (boolean): perform mean normalization
        flip_channels (boolean): flip channel order (ie. load rgb as bgr)
        blocksize (tuple): blocksize (w,h)
        use_channels (array): channels to be used
        normalization_mean (ndarray): mean per channel (required if normalize=true)
        flatten (bool): Flatten block dimensions (no distinction between rows)
    Return:
        Array of image blocks
    """
    image = rasterio_utils.load_image_data(filename)
    if use_channels:
        image = image[:, :, use_channels]  # slice channels

    if normalize:
        assert normalization_mean is not None, "No normalization mean given"
        image = normalize_image(image, normalization_mean)

    # take center crop to avoid view_as_blocks fail when image dimensions are
    # not exactly multiples of block size. E.g. images 6001x6000 with block
    # size 256 yields a cropped image of 5888x5888 pixels (256 * 23)
    # x = crop(x, [ / 2, x.shape[1] / img_cols / 2])
    # x = crop_blocksize(image, blocksize)
    patches = image_utils.image_as_windows(image,
                                           blocksize,
                                           overlap=overlap,
                                           flatten=flatten)

    return patches


def load_labels_as_blocks(filename, n_labels, blocksize=(32, 32)):
    """Load label image as blocks of given blocksize.

    Args:
        filename (string): Filename, expects unit8, where values are integer
                           indices of the classes.
        n_labels (integer): Number of labels
        blocksize (tuple): blocksize (w,h)
    Return:
        Array of image blocks
    """
    img_rows, img_cols = blocksize
    y = rasterio_utils.load_image_data(filename)
    y = np.squeeze(y, axis=-1)
    y = decompress_labels(y, n_labels)

    y = crop_blocksize(y, blocksize)

    y = view_as_blocks(y, (img_rows, img_cols, y.shape[-1]))

    y = y.reshape((-1, y.shape[3] * y.shape[4], y.shape[5]))

    return y


def get_label_mask(label_image, label_index):
    label_mask = label_image[..., label_index]
    return label_mask
