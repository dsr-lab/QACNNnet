import tensorflow as tf


class HighwayLayer(tf.keras.layers.Layer):

    # input-independent initialization
    def __init__(self, activation=None, dropout=0.1, kernel_size=1):
        super(HighwayLayer, self).__init__()

        # Class variables
        self.kernel_size = kernel_size
        self.activation = activation

        # Layers
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.transform = None
        self.conv = None

    # input-dependent initialization
    def build(self, input_lenght):
        n_filters = input_lenght[-1]
        self.transform = tf.keras.layers.Conv1D(n_filters, self.kernel_size, activation='sigmoid')
        self.conv = tf.keras.layers.Conv1D(n_filters, self.kernel_size, activation=self.activation)

    # forward computation
    def call(self, inputs):
        t = self.transform(inputs)

        h = self.conv(inputs)
        h = self.dropout(h)

        result = h * t + inputs * (1.0 - t)
        return result
