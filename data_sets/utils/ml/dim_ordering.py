import keras.backend as backend


def dim_correction_decorator(generator_fn):
    """
    All generators in the framework produce data in tf dimension ordering. This decorator should be placed over
    the generator functions producing data for training to correct dimension ordering.
    It requires 1st element of tuple returned by generator to be image batch.
    """

    def decorator(*args, **kwargs):
        generator = generator_fn(*args, **kwargs)
        for batch in generator:
            yield (correct_model_input(batch[0]),) + batch[1:]

    return decorator


def correct_model_input(input_batch):
    """
    Corrects dimension odering of input_batch of images such that it matches model input.
    :param model: A valid keras model.
    :param input_batch: Input image batch.
    :return: Input batch with correct dimension ordering.
    """
    if backend.image_dim_ordering() == 'th':
        return input_batch.transpose(0, 3, 1, 2)
    else:
        return input_batch


def get_block_size(model):
    """
    Returns size of model input (h,w).
    :param model:  A valid keras model.
    :return: (h,w) of model input.
    """
    if backend.image_dim_ordering() == 'th':
        blocksize = model.input_shape[2:]
    else:
        blocksize = model.input_shape[1:3]
    return blocksize
