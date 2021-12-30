from unittest import TestCase

from context_query_attention import ContextQueryAttentionLayer
import numpy as np
from tensorflow.keras import initializers

class TestContextQueryAttentionLayer(TestCase):
    def test_build_similarity_matrix(self):

        # Arrange
        layer = ContextQueryAttentionLayer(2)
        layer.w.kernel_initializer = initializers.ones

        BATCH_SIZE = 1
        N_CONTEXT = 4
        N_QUERY = 2
        N_DIM = 2
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

        expected_result = np.zeros((BATCH_SIZE, N_CONTEXT * N_QUERY, N_DIM * 3))
        idx = 0
        for b in range(BATCH_SIZE):
            for c in context[b]:
                for q in query[b]:
                    c_q = np.concatenate((c, q))
                    c_dot_q = np.multiply(c, q)
                    expected_result[b, idx] = np.concatenate((c_q, c_dot_q))
                    idx += 1
        expected_result = layer.w(expected_result)
        expected_result = np.reshape(expected_result, (BATCH_SIZE, N_CONTEXT, N_QUERY))

        # Act
        result = layer.build_similarity_matrix(context, query)


        # Assert
        np.testing.assert_array_equal(result, expected_result)
