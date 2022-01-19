import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from encoding import positional_encoding, stochastic_dropout




def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.compat.v1.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        tf.math.log(float(max_timescale) / float(min_timescale)) /
            (tf.compat.v1.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.compat.v1.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.math.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.experimental.numpy.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


class EncodingLayer(layers.Layer):

    def __init__(self,
                 d_model: int,
                 kernel_size: int,
                 n_conv_layers: int,
                 n_heads: int,
                 survival_prob: float,
                 block_num: int,
                 dropout_rate: float,
                 l2_rate: float):
        '''
        Parameters:
        -----------
        d_model: int
            Dimensions of the model (input and output of each layer)
        kernel_size: int
            Kernel size to use in convolutions
        n_conv_layers: int
            Number of Convolutional layers
        n_heads: int
            Number of heads for MultiHeadAttention
        survival_prob: float
            Survival probability of a layer for stochastic dropout.
        l2_rate: float
            L2 value used in L2 regularization
        block_num: int
            Index of the current encoding layer
        '''

        super(EncodingLayer, self).__init__()

        self.d_model = d_model
        self.n_layers = n_conv_layers + 2
        self.survival_prob = survival_prob
        self.block_num = block_num

        # Regularizer
        l2 = None if l2_rate == 0.0 else tf.keras.regularizers.l2(l2_rate)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.conv_layer_params = {
            "filters": d_model,
            "kernel_size": kernel_size,
            "padding": "same",  # necessary for residual blocks
            "data_format": "channels_last",
            "depthwise_regularizer": l2,
            "pointwise_regularizer": l2,
            # "activity_regularizer": l2,
            "bias_regularizer": l2,
            "activation": "relu",
            "kernel_initializer": tf.keras.initializers.HeNormal()
        }

        self_attention_layer_params = {
            "num_heads": n_heads,
            "key_dim": d_model,
            "kernel_regularizer": l2,
            # "activity_regularizer": l2,
            "bias_regularizer": l2
        }

        # feed_forward_layer_params = {
        #     "units": d_model,
        #     "activation": "relu",
        #     "kernel_regularizer": l2,
        #     # "activity_regularizer": l2,
        #     "bias_regularizer": l2
        # }

        self.norm_layers = [layers.LayerNormalization() for _ in range(self.n_layers)]

        self.conv_layers = [layers.SeparableConv1D(**self.conv_layer_params) for _ in range(n_conv_layers)]

        self.self_attention_layer = layers.MultiHeadAttention(**self_attention_layer_params)

        # self.feed_forward_layer = layers.Dense(**feed_forward_layer_params)  # Is one layer enough?

        # TODO: create a dictionary like the other layers
        self.ff1 = layers.Conv1D(d_model, 1, activation='relu',
                                 kernel_regularizer=l2, bias_regularizer=l2,
                                 kernel_initializer=tf.keras.initializers.HeNormal())
        self.ff2 = layers.Conv1D(d_model, 1, activation=None,
                                 kernel_regularizer=l2, bias_regularizer=l2)

    def compute_attention_mask(self, mask):

        n = mask.shape[1]

        horizontal_mask = layers.RepeatVector(n)(mask)

        reshaped_mask = layers.Reshape((n, 1))(mask)
        vertical_mask = layers.Concatenate(axis=-1)([reshaped_mask for _ in range(n)])
        vertical_mask = layers.Reshape((n, n))(vertical_mask)

        attention_mask = horizontal_mask & vertical_mask
        return attention_mask

    def apply_layer(self,
                    x: tf.Tensor,
                    layer_num: int,
                    layer: layers.Layer,
                    training: bool,
                    attention_mask=None,
                    feed_forward=False) -> tf.Tensor:
        '''
        Check whether a layer should be used or not (stochastic dropout), then:
        -Return the input as the output if the layer should not be used;
        -Return layer(layer_normalization(x)) + x if the layer should be used.

        Parameters:
        ----------
        x: tf.Tensor
            Input of the layer
        layer_num: int
            Number of the current layer in the block
        layer: layers.Layer
            Layer to be applied (one between SeparableConv1D, MultiHeadAttention or Dense)
        training: bool
            True if the model is training, False otherwise
        attention_mask: tf.Tensor
            Boolean mask used for MultiHeadAttention

        Returns:
        --------
        x: tf.Tensor
            Output of the layer with stochastic dropout.
        '''

        keep = stochastic_dropout.keep_layer(self.n_layers, layer_num, self.survival_prob) if training else True

        if keep:
            can_apply_residual_block = x.shape[-1] == self.d_model

            norm_x = self.norm_layers[layer_num](x)

            # Apply dropout
            if (layer_num % 2 == 0 and type(layer) == layers.SeparableConv1D) or \
                    type(layer) == layers.MultiHeadAttention or \
                    feed_forward:
                norm_x = self.dropout(norm_x)

            # Different behaviour if the current layer is the feedforward block
            if feed_forward:
                f_x = self.ff1(norm_x)
                f_x = self.ff2(f_x)
            elif type(layer) != layers.MultiHeadAttention:
                f_x = layer(norm_x)
            else:
                f_x = layer(norm_x, norm_x,attention_mask=attention_mask)

            # Residual block
            if can_apply_residual_block:
                f_x = self.dropout(f_x)
                return f_x + x
            else:
                f_x = self.dropout(f_x)
                return f_x
        else:
            # tf.print("not keeping layer: ", layer_num, self.norm_layers[layer_num])
            return x

    def call(self, x, training, mask=None):

        '''
        Override of keras.layers.Layer method: computes the output of an Encoding Layer:
        1. Apply positional_encoding if required;
        2. Apply a series of separable convolutional layers;
        3. Apply one self-attention layer;
        4. Apply one dense layer.

        The input of each layer is the residual block using a normalization layer to the output of the previous layer.
        '''
        embedding_size = x.shape[2]
        maximum_position_encoding = x.shape[1]
        pos_encoding = positional_encoding.get_encoding(maximum_position_encoding,
                                                        embedding_size)

        attention_mask = self.compute_attention_mask(mask)

        current_layer_num = 0

        # 1. Apply positional encoding if it is the first block
        if self.block_num == 0:
            # seq_len = tf.shape(x)[1]
            # x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32)) #Necessary?

            x = add_timing_signal_1d(x)
            # x += pos_encoding[:, :seq_len, :]

            print()




        # 2. Convolution block
        for conv_layer in self.conv_layers:
            x = self.apply_layer(x, current_layer_num, conv_layer, training)
            current_layer_num += 1

        # 3. Self-attention block
        x = self.apply_layer(x, current_layer_num, self.self_attention_layer, training, attention_mask=attention_mask)
        current_layer_num += 1

        # 4. Feed-forward block
        # x = self.apply_layer(x, current_layer_num, self.feed_forward_layer, training)
        x = self.apply_layer(x, current_layer_num, None, training, feed_forward=True)

        return x


class EncoderLayer(layers.Layer):
    def __init__(self,
                 d_model: int,
                 kernel_size: int,
                 n_conv_layers: int,
                 n_heads: int,
                 survival_prob: float,
                 n_blocks: int,
                 dropout_rate: float,
                 l2_rate: float):

        '''
        Parameters:
        -----------
        d_model: int
            Dimensions of the model (input and output of each layer)
        kernel_size: int
            Kernel size to use in convolutions
        n_conv_layers: int
            Number of Convolutional layers
        n_heads: int
            Number of heads for MultiHeadAttention
        survival_prob: float
            Survival probability of a layer for stochastic dropout.
        l2_rate: float
            L2 value used in L2 regularization
        n_blocks: int
            Number of encoding layers to stack
        '''

        super(EncoderLayer, self).__init__()

        self.encoding_blocks = [EncodingLayer(d_model,
                                              kernel_size,
                                              n_conv_layers,
                                              n_heads,
                                              survival_prob,
                                              i,
                                              dropout_rate,
                                              l2_rate) for i in range(n_blocks)]

    def call(self, x, training, mask=None):
        """
        Override of keras.layers.Layer method: computes the output of an Encoder Layer,
        by computing the output of all the stacked Encoding layers.
        """

        for encoding_block in self.encoding_blocks:
            x = encoding_block(x, training=training, mask=mask)

        return x
