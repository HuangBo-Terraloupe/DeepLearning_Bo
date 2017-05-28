import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy import misc

def get_discrete_sliding_window_boxes(img_size, bbox_size, stride=None, offset=None, boundary=None,
                                      is_boundary_center=False, is_padding=False):
    """Computes discrete rasterized bounding box locations.

    Notes: The case where boundaries count towards bounding box edges, with padding enabled is not supported.

    Args:
        img_size: Tuple of (image_width, image_height)
        bbox_size: Tuple of (bounding_box_width, bounding_box_height)
        stride: If of instance int, stride is a tuple of pixel strides for extracting bounding box positions
                in (x, y) directions. If of instance float with values >= 0. and < 1. it is interpreted as percentage
                of overlap in (x, y) directions.
        offset: Tuple of (x, y), shifting the starting location for grabbing bounding boxes to
                (right, down). The values have to be >= 0 and < their corresponding stride.
        boundary: List with four numbers specifying the distances to their corresponding image borders
                  [left, top, right, bottom] (inclusively)
        is_boundary_center: If True, boundary parameter relates to bounding box centers. If False, it relates to
                            bounding box edges.
        is_padding: determines if bounding boxes are allowed to reach outside of the confines of the image size

    Returns:
        object: Returns a list of lists of bounding box coordinates specified by (ul_x, ul_y, lr_x, lr_y))

    """
    # Checking if image size is valid
    assert (len(img_size) == 2)
    for ims in img_size:
        assert (isinstance(ims, int))
        assert (ims > 1)

    # Checking bounding box specifications
    assert (len(bbox_size) == 2)
    for i in [0, 1]:
        assert (isinstance(bbox_size[i], int))
        if not is_padding:
            assert (0 < bbox_size[i] < img_size[i])

    # Checking stride and converting percentage to pixel stride if necessary
    if stride is None:
        stride = [1, 1]
    assert (len(stride) == 2)
    if isinstance(stride[0], int) and isinstance(stride[1], int):
        for s in stride:
            assert (s > 0)
    else:
        assert(isinstance(stride[0], float) and isinstance(stride[1], float))
        for s in stride:
            assert (0. <= s < 1.)
        tmp = [0, 0]
        tmp[0] = int(np.round(bbox_size[0] * (1. - stride[0])))
        tmp[1] = int(np.round(bbox_size[1] * (1. - stride[1])))
        for t in tmp:
            assert (isinstance(t, int))
            assert (t > 0)
        stride = tmp

    # Checking if offset is valid
    if offset is None:
        offset = [0, 0]
    assert (len(offset) == 2)
    for i in [0, 1]:
        assert (isinstance(offset[i], int))
        assert (0 <= offset[i] < stride[i])

    # Checking boundary specifications
    if boundary is None:
        boundary = [0, 0, 0, 0]
    assert (len(boundary) == 4)
    for b in boundary:
        assert (isinstance(b, int))
        assert (0 <= b)

    # Check if unsupported case
    if not is_boundary_center and is_padding:
        print "The case of boundaries counting towards edges with overlap is not supported! "
        assert()

    # Updating boundary if necessary
    if is_boundary_center and not is_padding:
        boundary[0] = max(np.floor(bbox_size[0] / 2.), boundary[0])
        boundary[1] = max(np.floor(bbox_size[1] / 2.), boundary[1])
        boundary[2] = max(np.floor(bbox_size[0] / 2.) + bbox_size[0] % 2, boundary[2])
        boundary[3] = max(np.floor(bbox_size[1] / 2.) + bbox_size[1] % 2, boundary[3])
    elif not is_boundary_center:
        boundary[2] += bbox_size[0]
        boundary[3] += bbox_size[1]

    # Compute relative shift from bounding box center to upper left corner
    shift_x = int(np.floor(bbox_size[0] / 2.))
    shift_y = int(np.floor(bbox_size[0] / 2.))

    # Generate raster coordinates for remaining valid region
    ul_coordinates = list(product(np.arange(boundary[0] + offset[0],
                                            img_size[0] - boundary[2] - 1 - ((stride[0] - offset[0]) % stride[0]) + 1,
                                            stride[0]),
                                  np.arange(boundary[1] + offset[1],
                                            img_size[1] - boundary[3] - 1 - ((stride[1] - offset[1]) % stride[1]) + 1,
                                            stride[1])))

    # Compute bounding box coordinates and shift bounding boxes relative to its center if necessary.
    if is_boundary_center:
        bbox_list = [(x - shift_x,
                      y - shift_y,
                      x - shift_x + bbox_size[0],
                      y - shift_y + bbox_size[1]) for x, y in ul_coordinates]
    else:
        bbox_list = [(x, y, x + bbox_size[0], y + bbox_size[1]) for x, y in ul_coordinates]

    return bbox_list


def crop_image(image, crop_size, offset=None, seed=None):

    """ Crops the image to a specific target size.
        if the cropping offset is not mentioned, then the crop is taken at random.

    Args:
        image: numpy array of the image ( order h X w X c or h X w i.e. channels at the end )
        crop_size: The crop size required
        offset: the top left corner from where the cropping is needed
        seed: seed

    Returns: cropped image

    """
    h, w = image.shape[0:2]

    if offset:
        assert offset[0] + (crop_size[0]-1) < h, "Cannot crop {} area from {} image".format(offset, image.shape)
        assert offset[1] + (crop_size[0]-1) < w, "Cannot crop {} area from {} image".format(offset, image.shape)

    else:
        # Randomly crops the image to a specific target size
        if seed:
            np.random.seed(seed)
        offset = [np.random.randint(0, h - (crop_size[0]-1)), np.random.randint(0, w - (crop_size[1]-1))]

    if len(image.shape) > 2:
        image = image[offset[0]:offset[0] + crop_size[0], offset[1]:offset[1] + crop_size[1], :]
    else:
        image = image[offset[0]:offset[0] + crop_size[0], offset[1]:offset[1] + crop_size[1]]
        image = np.expand_dims(image, axis=-1)

    return image


# -----------------------------------    softmax to classify the label  -----------------------------------------------#

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

# -----------------------------------     sliding window to evaluate the whole image   -------------------------------#

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

def predict_single(model, img_patch):
    """
    Predicts the response maps of a single image map. Model should output the response map of same
    size as img_patch.
    :param model: A keras model.
    :param img_patch: Image patch of form (batch_size, h, w, channels).
    :return:
    """
    patch_size = model.input_shape[1:3]
    img_patch_norm = normalize_image_channelwise(img_patch)
    img_patch_norm = np.expand_dims(img_patch_norm, 0)
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


def discrete_matshow(data , nb_classes, label_dic):
    '''
    This function to show the output of segmentation with legend
    :param data: the segmentation output of cnn
    :param nb_classes: number of classes
    :param label_dic: the name of label, is a list and the position is proportional to the pixel values
    :return: None, plot the label with legend
    '''

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    #cmap = plt.get_cmap('RdBu', nb_classes)

    bounds = np.linspace(0, nb_classes, nb_classes+1)

    mat = plt.matshow(data, cmap=cmap)
    cax = plt.colorbar(mat, spacing='proportional', ticks=np.arange(nb_classes), boundaries=bounds)

    cax.ax.set_yticklabels(label_dic)
    plt.show()

