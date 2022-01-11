import tensorflow as tf
from tensorflow.keras import layers

from input_embedding.char_embedding_layer import CharEmbeddingLayer
from input_embedding.highway_layer import HighwayLayer
from input_embedding.word_embedding_layer import WordEmbeddingLayer


class InputEmbeddingLayer(tf.keras.layers.Layer):

    # input-independent initialization
    def __init__(self,
                 w_emb_size, w_pretrained_weights, w_vocab_size, w_n_special_tokens,
                 c_emb_size, c_vocab_size, c_conv_kernel_size, n_highway_layers,
                 dropout_rate=0.0):
        super(InputEmbeddingLayer, self).__init__()

        # Layers
        self.char_embedding = CharEmbeddingLayer(
            c_emb_size, c_vocab_size, c_conv_kernel_size, dropout_rate=dropout_rate/2)
        self.word_embedding = WordEmbeddingLayer(
            w_emb_size, w_pretrained_weights, w_vocab_size, w_n_special_tokens, dropout_rate=dropout_rate)

        self.highway_layers = []
        for i in range(n_highway_layers):
            self.highway_layers.append(HighwayLayer())

    # input-dependent initialization
    def build(self, input_lenght):
        pass

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

        for highway in self.highway_layers:
            final_emb = highway(final_emb)

        return final_emb, mask
