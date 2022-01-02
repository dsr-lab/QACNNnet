import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from context_query_attention import ContextQueryAttentionLayer
from encoding.encoder import EncoderLayer
from input_embedding.input_embedding_layer import InputEmbeddingLayer
from model_output import OutputLayer


class QACNNnet(tf.keras.Model):

    def __init__(self, input_embedding_params, embedding_encoder_params, conv_layer_params, model_encoder_params):
        super(QACNNnet, self).__init__()

        self.embedding = InputEmbeddingLayer(**input_embedding_params)
        self.embedding_encoder = EncoderLayer(**embedding_encoder_params)
        self.context_query_attention = ContextQueryAttentionLayer()
        self.model_encoder = EncoderLayer(**model_encoder_params)
        self.conv_1d = layers.SeparableConv1D(**conv_layer_params)
        self.model_output = OutputLayer()

    def call(self, inputs, training=False):
        assert len(inputs) == 4

        words_context = inputs[0]
        characters_context = inputs[1]
        words_query = inputs[2]
        characters_query = inputs[3]

        # 1. Embedding blocks
        context_embedded, context_mask = self.embedding([words_context, characters_context])
        query_embedded, query_mask = self.embedding([words_query, characters_query])

        # 2. Embedding encoder blocks
        context_encoded = self.embedding_encoder(context_embedded, training=training, mask=context_mask)
        query_encoded = self.embedding_encoder(query_embedded, training=training, mask=query_mask)

        # 3. Context-query attention block
        attention_output = self.context_query_attention([context_encoded, query_encoded], [context_mask, query_mask])
        attention_output = self.conv_1d(attention_output)

        # 4. Model encoder blocks
        m0 = self.model_encoder(attention_output, training=training, mask=context_mask)
        m1 = self.model_encoder(m0, training=training, mask=context_mask)
        m2 = self.model_encoder(m1, training=training, mask=context_mask)

        # 5. Output block
        output = self.model_output([m0, m1, m2], mask=context_mask)

        return output

    def model(self, max_context_words, max_query_words, max_chars):
        context_words_input = tf.keras.Input(shape=(max_context_words), name="context words")
        context_characters_input = tf.keras.Input(shape=(max_context_words, max_chars), name="context characters")
        query_words_input = tf.keras.Input(shape=(max_query_words), name="query words")
        query_characters_input = tf.keras.Input(shape=(max_query_words, max_chars), name="query characters")

        inputs = [context_words_input, context_characters_input, query_words_input, query_characters_input]

        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
