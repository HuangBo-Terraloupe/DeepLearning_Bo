def load_and_transfer(model_file, weights_file):
    """ Transfer weights of a model trained on multiple GPU's to a model running on single GPU.

    Args:
        model_file : Non-Parallelized version of JSON model file
        weights_file : Model weights saved in parallelized training
    Returns: A keras model capable of running on single GPU

    """
    from keras.models import model_from_json
    import h5py

    model = model_from_json(open(model_file, "rb").read())
    f = h5py.File(weights_file, mode='r')
    w = f["model_weights"]["model_1"]
    for i, layer in enumerate(model.layers):
        layer_weights = layer.weights
        weights_to_set = []
        for params in layer_weights:
            weight_name = params.name
            saved_weights = w[weight_name].value
            weights_to_set.append(saved_weights)
        model.layers[i].set_weights(weights_to_set)

    return model