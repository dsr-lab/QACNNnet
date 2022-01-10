import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from encoding import positional_encoding, stochastic_dropout


class EncodingLayer(layers.Layer):
    def __init__(self,
                 d_model: int,
                 kernel_size: int,
                 n_conv_layers: int,
                 n_heads: int,
                 survival_prob: float,
                 l2_value: float,
                 block_num: int):
        """
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
        l2_value: float
            L2 value used in L2 regularization
        block_num: int
            Index of the current encoding layer
        """

        super(EncodingLayer, self).__init__()

        self.d_model = d_model
        self.n_layers = n_conv_layers + 2
        self.survival_prob = survival_prob
        self.block_num = block_num
        self.l2_decay = regularizers.l2(l2_value)

        self.conv_layer_params = {
            "filters": d_model,
            "kernel_size": kernel_size,
            "padding": "same",  # necessary for residual blocks
            "data_format": "channels_last",
            "kernel_regularizer": self.l2_decay
        }

        self_attention_layer_params = {
            "num_heads": n_heads,
            "key_dim": d_model,
            "kernel_regularizer": self.l2_decay
        }

        feed_forward_layer_params = {
            "units": d_model,
            "activation": "tanh",  # or Relu?
            #"activation": "relu",
            "kernel_regularizer": self.l2_decay
        }

        self.norm_layers = [layers.LayerNormalization() for _ in range(self.n_layers)]

        self.conv_layers = [layers.SeparableConv1D(**self.conv_layer_params) for _ in range(n_conv_layers)]

        self.self_attention_layer = layers.MultiHeadAttention(**self_attention_layer_params)

        # self.feed_forward_layer = layers.Dense(**feed_forward_layer_params)  # Is one layer enough?

        # self.ff1 = layers.Conv1D(d_model, 1, activation='relu')
        self.ff1 = layers.Conv1D(d_model, 1, activation='tanh')
        self.ff2 = layers.Conv1D(d_model, 1, activation=None)

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
            if feed_forward:
                f_x = self.ff1(norm_x)
                f_x = self.ff2(f_x)
            else:
                f_x = layer(norm_x) if type(layer) != layers.MultiHeadAttention else layer(norm_x, norm_x,
                                                                                       attention_mask=attention_mask)

            if can_apply_residual_block:
                return f_x + x
            else:
                return f_x
        else:
            return x

    def call(self, x, training, mask=None):

        """
        Override of keras.layers.Layer method: computes the output of an Encoding Layer:
        1. Apply positional_encoding if required;
        2. Apply a series of separable convolutional layers;
        3. Apply one self-attention layer;
        4. Apply one dense layer.

        The input of each layer is the residual block using a normalization layer to the output of the previous layer.
        """
        maximum_position_encoding = x.shape[1]
        embedding_size = x.shape[2]
        # TODO: optimizable?
        pos_encoding = positional_encoding.get_encoding(maximum_position_encoding,
                                                        embedding_size)

        attention_mask = self.compute_attention_mask(mask)

        current_layer_num = 0

        # 1. Apply positional encoding if it is the first block
        if self.block_num == 0:
            seq_len = tf.shape(x)[1]
            # x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32)) #Necessary?
            x += pos_encoding[:, :seq_len, :]

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
                 l2_value: float,
                 n_blocks: int):
        """
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
        l2_value: float
            L2 value used in L2 regularization
        n_blocks: int
            Number of encoding layers to stack
        """

        super(EncoderLayer, self).__init__()

        self.encoding_blocks = [EncodingLayer(d_model,
                                              kernel_size,
                                              n_conv_layers,
                                              n_heads,
                                              survival_prob,
                                              l2_value,
                                              i) for i in range(n_blocks)]

    def call(self, x, training, mask=None):
        """
        Override of keras.layers.Layer method: computes the output of an Encoder Layer,
        by computing the output of all the stacked Encoding layers.
        """

        for encoding_block in self.encoding_blocks:
            x = encoding_block(x, training=training, mask=mask)

        return x
