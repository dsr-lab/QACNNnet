import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class OutputLayer(layers.Layer):

    def __init__(self, l2_rate):
        '''
        Create the last block of the QACNNet.

        Parameters:
        -----------
        l2_rate: float
            The l2 rate.
            Passing 0.0 means that l2 regularization is not applied.
        '''

        super(OutputLayer, self).__init__()

        # Regularizer
        l2 = None if l2_rate == 0.0 else tf.keras.regularizers.l2(l2_rate)

        # Layers definition
        self.concatenate_layer = layers.Concatenate(axis=-1)
        self.stack = layers.Concatenate(axis=1)

        self.w1 = layers.Dense(units=1, use_bias=False,
                               kernel_regularizer=l2)
        self.w2 = layers.Dense(units=1, use_bias=False,
                               kernel_regularizer=l2)

        self.softmax_layer = layers.Softmax(axis=-1, dtype='float32')

    def compute_probabilities(self, input_1, input_2, start, mask):
        '''
        Compute start or end probabilities for each token in an input sequence.
        '''

        n_words = input_1.shape[1]

        # Concatenate outputs from previous layers
        concat = self.concatenate_layer([input_1, input_2])

        # Multiply to trainable variables (linear step)
        weighted = self.w1(concat) if start else self.w2(concat)

        # Reshape to the expected dimension necessary for applyng the softmax
        reshaped = layers.Reshape((n_words,))(weighted)

        # Get final probabilities ignoring the padding tokens
        softmaxed = self.softmax_layer(reshaped, mask=mask)

        return softmaxed

    def call(self, inputs, mask=None):
        assert len(inputs) == 3

        m0 = inputs[0]
        m1 = inputs[1]
        m2 = inputs[2]

        # Compute start and end probabilities for each token excluding the padding
        start_probabilities = self.compute_probabilities(m0, m1, True, mask)
        end_probabilities = self.compute_probabilities(m0, m2, False, mask)

        # Stack the probabilities to get a single output
        output = self.stack([start_probabilities, end_probabilities])
        output = layers.Reshape((2, m0.shape[1]), dtype='float32')(output)

        return output
