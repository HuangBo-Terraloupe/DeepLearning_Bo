import numpy as np
import rasterio


def write_raster_data(file_name, raster_data, profile, dtype=np.float32):
    """
    Writes raster data to file with profile.
    :param file_name: Name of output file.
    :param raster_data: Raster data of shape (bands, rows, columns)
    :param profile: Dict of raster profile.
    :param dtype: Data type of out file.
    """
    bands = raster_data.shape[0]
    out_profile = dict(profile)
    new_options = {'count': bands, 'height': raster_data.shape[1], 'width': raster_data.shape[2],
                   'compress': 'lzw', 'dtype': dtype}
    out_profile.update(new_options)
    out_profile.pop('transform')  # depricated
    with rasterio.open(file_name, 'w', **out_profile) as dst:
        for i in range(bands):
            dst.write(raster_data[i].astype(dtype), i + 1)


def load_image_data(filename, window=None, last_dim_channel=True):
    """
    Loads image using raster io.
    :param filename: Path of file.
    :param window: Region of raster to be read. in format ((minx, maxx), (miny,maxy)) otherwise whole image.
    :param flip_channels: If true last dimension will be channel.
    :return: Raster data of shape (h, w, n_channels) if last_dim_channel else (n_channels, h, w)
    """
    with rasterio.open(filename) as src:
        if window:
            image_data = src.read(window=window)
        else:
            image_data = src.read()
    if last_dim_channel:
        image_data = image_data.transpose(1, 2, 0)
    return image_data
