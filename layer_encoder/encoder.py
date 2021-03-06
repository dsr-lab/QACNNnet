import tensorflow as tf
from tensorflow.keras import layers

from layer_encoder import positional_encoding, stochastic_dropout


class EncodingLayer(layers.Layer):

    def __init__(self,
                 d_model,
                 kernel_size,
                 n_conv_layers,
                 n_heads,
                 survival_prob,
                 block_num,
                 dropout_rate,
                 l2_rate):
        '''
        Initialize a single encoder block.

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
        block_num: int
            Index of the current layer_encoder layer
        dropout_rate: float
            The dropout rate.
            Passing 0.0 means that dropout is not applied.
        l2_rate: float
            The l2 rate.
            Passing 0.0 means that l2 regularization is not applied.

        '''

        super(EncodingLayer, self).__init__()

        self.d_model = d_model
        self.n_layers = n_conv_layers + 2  # +2 takes into account self-attention and Dense layers
        self.survival_prob = survival_prob
        self.block_num = block_num

        # Regularizer
        l2 = None if l2_rate == 0.0 else tf.keras.regularizers.l2(l2_rate)

        # Layers parameters
        self.conv_layer_params = {
            "filters": d_model,
            "kernel_size": kernel_size,
            "padding": "same",  # necessary for residual blocks
            "data_format": "channels_last",  # channels are represented by the last dimension
            "depthwise_regularizer": l2,
            "pointwise_regularizer": l2,
            "bias_regularizer": l2,
            "activation": "relu",
            "kernel_initializer": tf.keras.initializers.HeNormal()
        }

        self_attention_layer_params = {
            "num_heads": n_heads,
            "key_dim": d_model,
            "kernel_regularizer": l2,
            "bias_regularizer": l2
        }

        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Norm layers
        self.norm_layers = [layers.LayerNormalization() for _ in range(self.n_layers)]

        # Conv layers
        self.conv_layers = [layers.SeparableConv1D(**self.conv_layer_params) for _ in range(n_conv_layers)]

        # Self attention
        self.self_attention_layer = layers.MultiHeadAttention(**self_attention_layer_params)

        # Feed-forward
        self.ff1 = layers.Conv1D(d_model, 1, activation='relu',
                                 kernel_regularizer=l2, bias_regularizer=l2,
                                 kernel_initializer=tf.keras.initializers.HeNormal())
        self.ff2 = layers.Conv1D(d_model, 1, activation=None,
                                 kernel_regularizer=l2, bias_regularizer=l2)

    def compute_attention_mask(self, mask):
        '''
        Compute the mask that will be used in the self-attention layer,
        in order to avoid to consider <PAD> elements.
        '''

        n = mask.shape[1]

        # Repeat the vector-mask along the first axis
        horizontal_mask = layers.RepeatVector(n)(mask)

        # Repeat the vector-mask along the second axis
        reshaped_mask = layers.Reshape((n, 1))(mask)
        vertical_mask = layers.Concatenate(axis=-1)([reshaped_mask for _ in range(n)])
        vertical_mask = layers.Reshape((n, n))(vertical_mask)

        # Combine horizontal and vertical mask to get the desired matrix-mask
        attention_mask = horizontal_mask & vertical_mask

        return attention_mask

    def apply_layer(self,
                    x,
                    layer_num,
                    layer,
                    training,
                    attention_mask=None,
                    feed_forward=False):
        '''
        Check whether a layer should be used or not (stochastic dropout), then:
         - Return the input as the output if the layer is dropped out
         - Return layer (layer_normalization(x)) + x if the layer is mantained.

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
        feed_forward: bool
            Flag used for doing a different operation when the last feed-forward
            block is reached by the inputs

        Returns:
        --------
        x: tf.Tensor
            Output of the layer with stochastic dropout.
        '''

        # Decides whether to disable or not a layer using stochastic dropout
        keep = stochastic_dropout.keep_layer(self.n_layers, layer_num, self.survival_prob) if training else True

        if keep:
            can_apply_residual_block = x.shape[-1] == self.d_model

            # Apply layer normalization
            norm_x = self.norm_layers[layer_num](x)

            # Apply dropout
            if (layer_num % 2 == 0 and type(layer) == layers.SeparableConv1D) or \
                    type(layer) == layers.MultiHeadAttention or \
                    feed_forward:
                norm_x = self.dropout(norm_x)

            # Different behaviour if the current layer is the feed-forward block
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
            return x

    def call(self, x, training, mask=None):
        '''
        Computes the output of an Encoding Layer:
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

        # 1. Apply positional layer_encoder if it is the first block
        if self.block_num == 0:
            seq_len = tf.shape(x)[1]
            x += pos_encoding[:, :seq_len, :]

        # 2. Convolution block
        for conv_layer in self.conv_layers:
            x = self.apply_layer(x, current_layer_num, conv_layer, training)
            current_layer_num += 1

        # 3. Self-attention block
        x = self.apply_layer(x, current_layer_num, self.self_attention_layer, training, attention_mask=attention_mask)
        current_layer_num += 1

        # 4. Feed-forward block
        x = self.apply_layer(x, current_layer_num, None, training, feed_forward=True)

        return x


class EncoderLayer(layers.Layer):
    def __init__(self,
                 d_model,
                 kernel_size,
                 n_conv_layers,
                 n_heads,
                 survival_prob,
                 n_blocks,
                 dropout_rate,
                 l2_rate):

        '''
        Initialize an EncoderLayer, that could be composed by multiple EncodingLayers.

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
        block_num: int
            Index of the current layer_encoder layer
        dropout_rate: float
            The dropout rate.
            Passing 0.0 means that dropout is not applied.
        l2_rate: float
            The l2 rate.
            Passing 0.0 means that l2 regularization is not applied.
        '''

        super(EncoderLayer, self).__init__()

        # Stack n layer_encoder blocks to form the Encoder layer.
        self.encoding_blocks = [EncodingLayer(d_model,
                                              kernel_size,
                                              n_conv_layers,
                                              n_heads,
                                              survival_prob,
                                              i,
                                              dropout_rate,
                                              l2_rate) for i in range(n_blocks)]

    def call(self, x, training, mask=None):
        '''
        Computes the output of an Encoder Layer, by computing the output of all the stacked Encoding layers.
        '''

        for encoding_block in self.encoding_blocks:
            x = encoding_block(x, training=training, mask=mask)

        return x
