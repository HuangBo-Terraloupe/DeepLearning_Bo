import numpy as np
import random


def zip_list_generator(array1, array2, n, class_weights=None):
    """ Infinitely generate a zipped pairs of length n lists,
        where the first part is taken from array1, and the second part
        is taken from array2.

        Also works for numpy arrays.

        Example:
            zip_list_generator([0,1,2,3], [4,5,6,7] 3)
            => [([0,1,2], [4,5,6]), ([3,1,2], [7,4,5]), ...]
    """

    i = 0
    l = len(array1)
    assert len(array1) == len(array2)
    while True:
        a = i % l
        b = a + n
        if b <= l:
            X = array1[a:b]
            y = array2[a:b]
        else:
            X = np.concatenate([array1[a:l], array1[0:b - l]])
            y = np.concatenate([array2[a:l], array2[0:b - l]])
        if class_weights is None:
            yield (X, y)
        else:
            labels = y.argmax(axis=2)
            func = np.vectorize(class_weights.get)
            sample_weights = func(labels)
            yield (X, y, sample_weights)
        i += n


def partition_list(array, ratios, randomize=True, seed=None):
    ratios = np.array(ratios, dtype=float)
    ratios /= ratios.sum()
    ratios *= len(array)
    ratios = list(ratios.round().astype(int))

    array = list(array)

    if randomize:
        state = random.getstate()
        random.seed(seed)
        random.shuffle(array)
        random.setstate(state)

    partitions = []
    a = 0
    for r in ratios:
        b = a + r
        partitions.append(array[a:b])
        a = b

    return partitions
