import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#TODO: handle masks

class OutputLayer (layers.Layer):

    def __init__ (self):

        super(OutputLayer, self).__init__()

        self.concatenate_layer = layers.Concatenate(axis=-1)

        self.w1 = layers.Dense(units=1, use_bias=False)
        self.w2 = layers.Dense(units=1, use_bias=False)

        self.softmax_layer = layers.Softmax(axis=-1)

    def compute_probabilities (self, input_1, input_2, start):

        concat = self.concatenate_layer([input_1, input_2])

        weighted = self.w1(concat) if start else self.w2(concat)

        softmaxed = self.softmax_layer(weighted)

        n = tf.shape(input_1)[1]
        softmaxed = layers.Reshape((n)) (softmaxed)

        return softmaxed

    def call (self, inputs):

        assert len(input)==3

        m0 = inputs[0]
        m1 = inputs[1]
        m2 = inputs[2]

        start_probbabilities = self.compute_probabilities(m0, m1, True)
        end_probabilities = self.compute_probabilities(m0, m2, False)

        return start_probbabilities, end_probabilities
