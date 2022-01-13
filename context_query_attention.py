import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ContextQueryAttentionLayer (layers.Layer):

    def __init__(self, dropout_rate, l2_rate):

        super(ContextQueryAttentionLayer, self).__init__()

        # Regularizer
        l2 = None if l2_rate == 0.0 else tf.keras.regularizers.l2(l2_rate)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.w = layers.Dense(units=1, use_bias=False,
                              kernel_regularizer=l2)
                              # activity_regularizer=l2,
                              # bias_regularizer=l2


    def build_similarity_matrix(self, context, query):

        n = context.shape[1]
        m = query.shape[1]
        d_model = context.shape[-1]

        reshape_layer = layers.Reshape((n,m,d_model))
        mult_layer = layers.Multiply()
        stack_layer = layers.Concatenate(axis=-1)
        matrix_reshape_layer = layers.Reshape((n,m))

        conc_context = layers.Concatenate(axis=-1) ([context for _ in range(m)])
        conc_query = layers.Concatenate(axis=1) ([query for _ in range(n)])

        c_matrix = reshape_layer(conc_context)
        q_matrix = reshape_layer(conc_query)
        mult_matrix = mult_layer([q_matrix,c_matrix])

        similarity_matrix = stack_layer([q_matrix,c_matrix,mult_matrix])
        similarity_matrix = self.w(similarity_matrix)
        similarity_matrix = matrix_reshape_layer(similarity_matrix)

        return similarity_matrix

    def build_softmaxed_matrices (self, similarity_matrix, c_mask, q_mask):

        n = similarity_matrix.shape[1]
        m = similarity_matrix.shape[2]

        repeated_q_mask = layers.RepeatVector(n)(q_mask)

        reshaped_c_mask = layers.Reshape((n,1)) (c_mask)
        repeated_c_mask = layers.Concatenate(axis=-1) ([reshaped_c_mask for _ in range(m)])
        repeated_c_mask = layers.Reshape((n,m)) (repeated_c_mask)

        softmax_context_layer = layers.Softmax(axis=2)
        softmax_query_layer = layers.Softmax(axis=1)

        context_softmaxed_matrix = softmax_context_layer(similarity_matrix, mask=repeated_q_mask)
        query_softmaxed_matrix = softmax_query_layer(similarity_matrix, mask=repeated_c_mask)

        return context_softmaxed_matrix, query_softmaxed_matrix

    def build_attention_matrices (self, context, query, context_softmaxed_matrix, query_softmaxed_matrix):

        mult_layer = layers.Dot(axes=(2,1))

        context_to_query = mult_layer([context_softmaxed_matrix, query])

        transpose_layer = layers.Permute((2,1))
        transposed_query_softmaxed_matrix = transpose_layer(query_softmaxed_matrix)

        softmaxed_mult = mult_layer([context_softmaxed_matrix, transposed_query_softmaxed_matrix])

        query_to_context = mult_layer([softmaxed_mult,context])

        return context_to_query, query_to_context

    def build_output (self, context, context_to_query, query_to_context):

        mult_layer = layers.Multiply()
        conc_layer = layers.Concatenate(axis=-1)

        c_a = mult_layer([context, context_to_query])
        c_b = mult_layer([context, query_to_context])

        output = conc_layer([context, context_to_query, c_a, c_b])

        return output

    def call (self, inputs, masks):

        assert len(inputs)==2
        assert len(masks)==2

        context = inputs[0]
        context = self.dropout(context)
        query = inputs[1]
        query = self.dropout(query)

        c_mask = masks[0]
        q_mask = masks[1]

        similarity_matrix = self.build_similarity_matrix(context, query)

        context_softmaxed_matrix, query_softmaxed_matrix = self.build_softmaxed_matrices(similarity_matrix, c_mask, q_mask)

        context_to_query, query_to_context = self.build_attention_matrices(context, query, context_softmaxed_matrix, query_softmaxed_matrix)

        output = self.build_output(context, context_to_query, query_to_context)

        return output
