import tensorflow as tf


class CharEmbeddingLayer(tf.keras.layers.Layer):

    # input-independent initialization
    def __init__(self, emb_size, vocab_size, conv_kernel_size):
        super(CharEmbeddingLayer, self).__init__()

        # Class variables
        self.emb_size = emb_size
        self.vocab_size = vocab_size

        # Layers
        self.conv_layer = tf.keras.layers.Conv1D(emb_size, conv_kernel_size, activation='relu')
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
        y = self.emb_layer(inputs)
        y = self.conv_layer(y)
        y = self._maxpool(y)

        return y

    @staticmethod
    def _maxpool(x):
        y = tf.math.reduce_max(x, axis=2)
        return y
