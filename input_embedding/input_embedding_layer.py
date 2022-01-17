import tensorflow as tf
from tensorflow.keras import layers

from input_embedding.char_embedding_layer import CharEmbeddingLayer
from input_embedding.highway_layer import HighwayLayer
from input_embedding.word_embedding_layer import WordEmbeddingLayer


class InputEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self,
                 w_emb_size, w_pretrained_weights, w_vocab_size, w_n_special_tokens,
                 c_emb_size, c_vocab_size, c_conv_kernel_size, c_conv_output_size,
                 n_highway_layers, dropout_rate, l2_rate, conv_input_projection_params):
        '''
        Create the first block of the QACNNet

        Parameters:
        -----------
        w_emb_size: int
            The word embedding size
        w_pretrained_weights: tf.tensor
            Weighting matrix to apply to the word embedding layer
        w_vocab_size: int
            The number of words in the vocaboulary
        w_n_special_tokens: int
            The number of tokens considered special (e.g., <UNK>)
        c_emb_size: int
            The character embedding size
        c_vocab_size: int
            The number of characters in the vocaboulary
        c_conv_kernel_size: int
            The convolution kernel size used when processing characters
        n_highway_layers: int
            The number of highway layers that are used for generating
            the output of this layer
        dropout_rate: float
            The dropout rate.
            Passing 0.0 means that dropout is not applied.
        l2_rate: float
            The l2 rate.
            Passing 0.0 means that l2 regularization is not applied.
        '''

        super(InputEmbeddingLayer, self).__init__()

        # Regularizer
        l2 = None if l2_rate == 0.0 else tf.keras.regularizers.l2(l2_rate)

        # Layers
        self.char_embedding = CharEmbeddingLayer(
            c_emb_size, c_vocab_size, c_conv_kernel_size, c_conv_output_size, dropout_rate=dropout_rate/2)
        self.word_embedding = WordEmbeddingLayer(
            w_emb_size, w_pretrained_weights, w_vocab_size, w_n_special_tokens, dropout_rate=dropout_rate)

        self.conv_1d = layers.SeparableConv1D(**conv_input_projection_params)
        self.highway_layers = []
        for i in range(n_highway_layers):
            self.highway_layers.append(HighwayLayer(dropout_rate, l2))

    # forward computation
    def call(self, inputs):

        assert len(inputs)==2

        w_inputs = inputs[0]
        c_inputs = inputs[1]

        w_emb, mask = self.word_embedding(w_inputs)
        c_emb = self.char_embedding(c_inputs)

        # mask = w_inputs != 0
        # mask2 = self.word_embedding.emb_layer.compute_mask(w_inputs).numpy()

        #final_emb = tf.keras.layers.concatenate([w_emb, c_emb], axis=2)
        final_emb = layers.Concatenate(axis=2)([w_emb, c_emb])

        final_emb = self.conv_1d(final_emb)

        for highway in self.highway_layers:
            final_emb = highway(final_emb)

        return final_emb, mask
