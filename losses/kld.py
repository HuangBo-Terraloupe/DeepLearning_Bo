import keras
import numpy as np
import tensorflow as tf
from keras.losses import kld


def kl_divergence(labels, prediction, epsilon=1e-7):

    # normalize to distribution
    prediction /= (tf.reduce_sum(prediction, axis=1, keep_dims=True) + epsilon)
    labels /= (tf.reduce_sum(labels, axis=1, keep_dims=True) + epsilon)
    result = tf.reduce_mean(tf.reduce_sum(labels * tf.log((labels / (prediction + epsilon)) + epsilon), axis=1))
    return result


# gt = np.random.rand(1, 10)
gt = np.ones([5, 10])
print('the ground truth is:', gt)
gt = tf.convert_to_tensor(gt)

# pre = np.random.rand(1, 10)
pre = np.zeros([5, 10])
print('the prediction is:', pre)
pre = tf.convert_to_tensor(pre)

sess = tf.InteractiveSession()
print('tensorflow kld:', kl_divergence(gt, pre).eval())
print('keras kld:', kld(gt, pre).eval())