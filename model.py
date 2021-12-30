#This class is responsible for assembling the final model and defining training and testing info

import tensorflow as tf
from tensorflow.keras import layers

def loss_function(y_true, y_pred):

    #y_true = (batch_size, 2, 1) or (batch_size, 2)
    #y_pred = (batch_size, 2, n_words)

    assert int(tf.shape(y_true)[1])==2
    assert int(tf.shape(y_pred)[1])==2

    batch_size = int(tf.shape(y_true)[0])

    y_true_start, y_true_end = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

    p1 = tf.gather(params=y_pred_start, indices=y_true_start, axis=-1, batch_dims=batch_size)
    p2 = tf.gather(params=y_pred_end, indices=y_true_end, axis=-1, batch_dims=batch_size)

    p1 = tf.reshape(p1,shape=(batch_size,1))
    p2 = tf.reshape(p2,shape=(batch_size,1))

    log_p1 = tf.math.log(p1)
    log_p2 = tf.math.log(p2)

    neg_log_p1 = tf.math.negative(log_p1)
    neg_log_p2 = tf.math.negative(log_p2)

    sum = neg_log_p1 + neg_log_p2

    mean = tf.reduce_mean(sum)

    return mean

#Test
true = tf.constant([[[0],[3]],[[1],[2]]])
pred = tf.constant([[[0.1,0.2,0.3,0.4],[0.2,0.35,0.65,0.78]],[[0.2,0.3,0.41,0.12],[0.13,0.45,0.78,0.21]]])

loss = loss_function(true,pred)
print(loss)
