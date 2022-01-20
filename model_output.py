import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class OutputLayer(layers.Layer):

    def __init__(self, l2_rate):
        super(OutputLayer, self).__init__()

        l2 = None if l2_rate == 0.0 else tf.keras.regularizers.l2(l2_rate)

        self.concatenate_layer = layers.Concatenate(axis=-1)
        self.stack = layers.Concatenate(axis=1)

        self.w1 = layers.Dense(units=1, use_bias=False,
                               kernel_regularizer=l2)
        self.w2 = layers.Dense(units=1, use_bias=False,
                               kernel_regularizer=l2)

        self.softmax_layer = layers.Softmax(axis=-1)

    def compute_probabilities(self, input_1, input_2, start, mask):
        # input1, input2 shapes = (BATCH_SIZE, N_WORDS, N_DIMS)

        n_words = input_1.shape[1]

        concat = self.concatenate_layer([input_1, input_2])  # (BATCH_SIZE, N_WORDS, N_DIMS * 2)

        weighted = self.w1(concat) if start else self.w2(concat)  # (BATCH_SIZE, N_WORDS, 1)

        reshaped = layers.Reshape((n_words,))(weighted)  # (BATCH_SIZE, N_WORDS)

        softmaxed = self.softmax_layer(reshaped, mask=mask)  # (BATCH_SIZE, N_WORDS)

        return softmaxed

    def call(self, inputs, mask=None):
        assert len(inputs) == 3

        m0 = inputs[0]
        m1 = inputs[1]
        m2 = inputs[2]

        start_probabilities = self.compute_probabilities(m0, m1, True, mask)
        end_probabilities = self.compute_probabilities(m0, m2, False, mask)

        output = self.stack([start_probabilities, end_probabilities])
        output = layers.Reshape((2, m0.shape[1]))(output)

        return output


'''
#Test
test = OutputLayer()
a = tf.constant(2,shape=(1,5,128),dtype=tf.float32)
b = tf.constant(5,shape=(1,5,128),dtype=tf.float32)
c = tf.constant(7,shape=(1,5,128),dtype=tf.float32)
_mask = tf.convert_to_tensor([[True,True,True,False,False]])
build = test([a,b,c], mask=_mask)
print(build)
'''
