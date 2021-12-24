import numpy as np
import tensorflow as tf


class CharEmbeddingLayer(tf.keras.layers.Layer):

    """
    The input that we should expect here is:
    [
        [[h, e, l, l, o], [w, o, r, l, d]],  # Sentece 1
        [[h, e, l, l, o], [w, o, r, l, d]],  # Sentece 2
        ...                                  # batch size
    ]
    """

    # Create a matrix with all the words s


    # input-independent initialization
    def __init__(self, emb_size):

        # Set variables
        self.emb_size = emb_size

        # Placeholders (Defined later)
        self.emb_layer = None

    # input-dependent initialization
    def build(self, input_lenght):
        self.emb_layer = tf.keras.layers.Embedding(
            self.vocab_size + 1,  # +1 for padding
            self.emb_size,
            input_length=input_lenght[-1],
            trainable=True,
            mask_zero=True
        )

    # forward computation
    def call(self, inputs):
        pass
