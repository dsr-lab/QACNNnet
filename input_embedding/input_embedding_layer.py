import tensorflow as tf

from input_embedding.char_embedding_layer import CharEmbeddingLayer
from input_embedding.highway_layer import HighwayLayer
from input_embedding.word_embedding_layer import WordEmbeddingLayer


class InputEmbeddingLayer(tf.keras.layers.Layer):

    # input-independent initialization
    def __init__(self,
                 w_emb_size, w_pretrained_weights, w_vocab_size, w_n_special_tokens,
                 c_emb_size, c_vocab_size, c_conv_output_size=96, c_conv_kernel_size=5,
                 n_highway_layers=2):
        super(InputEmbeddingLayer, self).__init__()

        # Layers
        self.char_embedding = CharEmbeddingLayer(
            c_emb_size, c_vocab_size, c_conv_output_size, c_conv_kernel_size)
        self.word_embedding = WordEmbeddingLayer(
            w_emb_size, w_pretrained_weights, w_vocab_size, w_n_special_tokens)

        self.highway_layers = []
        for i in range(n_highway_layers):
            self.highway_layers.append(HighwayLayer())

    # input-dependent initialization
    def build(self, input_lenght):
        pass

    # forward computation
    def call(self, w_inputs, c_inputs):
        w_emb = self.word_embedding(w_inputs)
        c_emb = self.char_embedding(c_inputs)

        final_emb = tf.keras.layers.concatenate([w_emb, c_emb], axis=2)

        for highway in self.highway_layers:
            final_emb = highway(final_emb)

        return final_emb
