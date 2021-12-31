from unittest import TestCase

from context_query_attention import ContextQueryAttentionLayer
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers

class TestContextQueryAttentionLayer(TestCase):

    @classmethod
    def setUpClass(cls):
        # cls.BATCH_SIZE = 1
        # cls.N_CONTEXT = 4
        # cls.N_QUERY = 2
        # cls.N_DIM = 2

        cls.BATCH_SIZE = 5
        cls.N_CONTEXT = 10
        cls.N_QUERY = 4
        cls.N_DIM = 5

        cls.c_mask = np.zeros((cls.BATCH_SIZE, cls.N_CONTEXT), dtype=bool)
        cls.q_mask = np.zeros((cls.BATCH_SIZE, cls.N_QUERY), dtype=bool)

        for b in range(cls.BATCH_SIZE):

            # Context mask
            c_inner_true = np.random.randint(1, cls.N_CONTEXT)  # at least one false
            c_inner_mask = np.ones(c_inner_true, dtype=bool)
            while len(c_inner_mask) < cls.N_CONTEXT:
                c_inner_mask = np.append(c_inner_mask, False)
            cls.c_mask[b] = c_inner_mask

            # Query mask
            q_inner_true = np.random.randint(1, cls.N_QUERY)  # at least one false
            q_inner_mask = np.ones(q_inner_true, dtype=bool)
            while len(q_inner_mask) < cls.N_QUERY:
                q_inner_mask = np.append(q_inner_mask, False)
            cls.q_mask[b] = q_inner_mask

    def test_build_similarity_matrix(self):

        # Arrange
        layer = ContextQueryAttentionLayer(2)
        layer.w.kernel_initializer = initializers.ones

        context = np.array(
            [
                [
                    [
                        1, 2
                    ],
                    [
                        3, 4
                    ],
                    [
                        5, 6
                    ],
                    [
                        7, 8
                    ]
                ]
            ]
        )

        query = np.array(
            [
                [
                    [
                        1, 2
                    ],
                    [
                        3, 4
                    ]
                ]
            ]
        )

        expected_result = np.zeros((self.BATCH_SIZE, self.N_CONTEXT * self.N_QUERY, self.N_DIM * 3))
        idx = 0
        for b in range(self.BATCH_SIZE):
            for c in context[b]:
                for q in query[b]:
                    c_q = np.concatenate((c, q))
                    c_dot_q = np.multiply(c, q)
                    expected_result[b, idx] = np.concatenate((c_q, c_dot_q))
                    idx += 1
        expected_result = layer.w(expected_result)
        expected_result = np.reshape(expected_result, (self.BATCH_SIZE, self.N_CONTEXT, self.N_QUERY))

        # Act
        result = layer.build_similarity_matrix(context, query)

        # Assert
        np.testing.assert_array_equal(result, expected_result)

    def test_build_softmaxed_matrices(self):
        # Arrange
        layer = ContextQueryAttentionLayer(self.N_DIM)
        context = np.random.rand(self.BATCH_SIZE, self.N_CONTEXT, self.N_DIM).astype(np.float32)
        query = np.random.rand(self.BATCH_SIZE, self.N_QUERY, self.N_DIM).astype(np.float32)

        c_processed_mask = np.expand_dims(self.c_mask, 2)
        q_processed_mask = np.expand_dims(self.q_mask, 1)

        # Act
        similarity_matrix = layer.build_similarity_matrix(context, query)
        context_softmaxed_matrix, query_softmaxed_matrix = layer.build_softmaxed_matrices(
            similarity_matrix, self.c_mask, self.q_mask)

        expected_context_softmaxed_matrix = tf.keras.layers.Softmax(axis=2)(similarity_matrix, mask=q_processed_mask)
        expected_query_softmaxed_matrix = tf.keras.layers.Softmax(axis=1)(similarity_matrix, mask=c_processed_mask)

        # BLOG IMPLEMENTATION
        S_, S_T = self.blog_softmaxed_matrices(similarity_matrix)

        # Assert
        np.testing.assert_array_equal(context_softmaxed_matrix, expected_context_softmaxed_matrix)
        np.testing.assert_array_equal(query_softmaxed_matrix, expected_query_softmaxed_matrix)
        np.testing.assert_array_equal(S_, context_softmaxed_matrix)
        np.testing.assert_array_equal(S_T, query_softmaxed_matrix)

    def test_build_attention_matrices(self):

        # Arrange
        layer = ContextQueryAttentionLayer(self.N_DIM)
        context = np.random.rand(self.BATCH_SIZE, self.N_CONTEXT, self.N_DIM).astype(np.float32)
        query = np.random.rand(self.BATCH_SIZE, self.N_QUERY, self.N_DIM).astype(np.float32)

        similarity_matrix = layer.build_similarity_matrix(context, query)
        context_softmaxed_matrix, query_softmaxed_matrix = layer.build_softmaxed_matrices(
            similarity_matrix, self.c_mask,self. q_mask)

        # Act
        context_to_query, query_to_context = layer.build_attention_matrices(
            context, query, context_softmaxed_matrix, query_softmaxed_matrix)

        expected_context_to_query = np.matmul(context_softmaxed_matrix, query, dtype=np.float32)
        expected_query_to_context = \
            np.matmul(
                np.matmul(context_softmaxed_matrix, np.transpose(query_softmaxed_matrix, (0, 2, 1)), dtype=np.float32),
                context, dtype=np.float32
            )

        # BLOG IMPLEMENTATION
        S_, S_T = self.blog_softmaxed_matrices(similarity_matrix)
        c2q = tf.linalg.matmul(S_, query)
        q2c = tf.linalg.matmul(tf.matmul(S_, np.transpose(S_T, (0, 2, 1))), context)

        # Assert
        np.testing.assert_almost_equal(context_to_query, expected_context_to_query)
        np.testing.assert_almost_equal(query_to_context, expected_query_to_context)
        np.testing.assert_almost_equal(expected_context_to_query, c2q)
        np.testing.assert_almost_equal(expected_query_to_context, q2c)

    def test_build_output(self):
        # Arrange
        layer = ContextQueryAttentionLayer(self.N_DIM)
        context = np.random.rand(self.BATCH_SIZE, self.N_CONTEXT, self.N_DIM).astype(np.float32)
        query = np.random.rand(self.BATCH_SIZE, self.N_QUERY, self.N_DIM).astype(np.float32)

        similarity_matrix = layer.build_similarity_matrix(context, query)
        context_softmaxed_matrix, query_softmaxed_matrix = layer.build_softmaxed_matrices(
            similarity_matrix, self.c_mask, self.q_mask)
        context_to_query, query_to_context = layer.build_attention_matrices(
            context, query, context_softmaxed_matrix, query_softmaxed_matrix)

        # Act
        output = layer.build_output(context, context_to_query, query_to_context)
        expexted_output = np.concatenate((context,
                                          context_to_query,
                                          np.multiply(context, context_to_query),
                                          np.multiply(context, query_to_context)),
                                         axis=-1).astype(np.float32)
        blog_output = self.blog_build_output(context, context_to_query, query_to_context)

        # Assert
        np.testing.assert_equal(output, expexted_output)
        np.testing.assert_equal(output, blog_output)

    @staticmethod
    def mask_logits(inputs, mask, mask_value=-1e30):
        mask = tf.cast(mask, tf.float32)
        result = inputs + mask_value * (1 - mask)
        return result

    def blog_softmaxed_matrices(self, similarity_matrix):
        mask_q = tf.expand_dims(self.q_mask, 1)
        S_ = tf.nn.softmax(self.mask_logits(similarity_matrix, mask=mask_q))  # context_softmaxed_matrix
        mask_c = tf.expand_dims(self.c_mask, 2)
        S_T = tf.nn.softmax(self.mask_logits(similarity_matrix, mask=mask_c), axis=1)  # query_softmaxed_matrix
        return S_, S_T

    @staticmethod
    def blog_build_output(c, c2q, q2c):
        f = [c, c2q, c * c2q, c * q2c]
        attention_outputs = [c, c2q, c * c2q, c * q2c]
        attention_outputs = np.concatenate(attention_outputs, axis=-1)
        return attention_outputs
