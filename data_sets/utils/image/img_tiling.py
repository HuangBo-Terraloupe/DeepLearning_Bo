import os.path


def parse_img_id(tiled_img_filename):
    """
    Parses img_id and row and column number from a tiled image
    filename.
    Args:
        tiled_img_filename: The filename of the tiled image

    Returns:
        (img_id, row_idx, col_idx)
    """

    filename = os.path.split(tiled_img_filename)[-1]
    file_identifier = os.path.splitext(filename)[0]

    split_loc = file_identifier.rindex('_')
    img_id = file_identifier[:split_loc]

    tile_id = file_identifier[split_loc + 1:]

    if 'x' in tile_id:
        row_idx, col_idx = tile_id.split('x')
    elif len(tile_id) == 2:
        row_idx, col_idx = tile_id
    else:
        raise Exception("Could not parse image tile ID from %s" % filename)
    return (img_id, int(row_idx), int(col_idx))


def get_img_id(image_filename):
    """
    Splits image filename into its base name and extension.
    :param image_name:
    :return: Image id and extension.
    """
    filename = os.path.split(image_filename)[-1]
    last_index = filename.rfind('.')
    image_id = filename[:last_index]
    ext = filename[last_index:]
    return image_id, ext
