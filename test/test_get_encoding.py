from unittest import TestCase
import numpy as np

from positional_encoding import get_encoding


class TestGet_encoding(TestCase):

    def test_call_getEncoding(self):
        # Arrange
        MAX_WORDS = 5
        EMB_SIZE = 10
        BATCH_SIZE = 1
        expected_result = np.zeros((BATCH_SIZE, MAX_WORDS, EMB_SIZE), dtype=np.float32)
        for b in range(BATCH_SIZE):
            for i in range(MAX_WORDS):
                for j in range(EMB_SIZE):
                    if j % 2 == 0:
                        expected_result[b, i, j] = np.sin(i / np.power(10000, 2 * j / EMB_SIZE))
                    else:
                        expected_result[b, i, j] = np.cos(i / np.power(10000, 2 * j / EMB_SIZE))

        # Act
        result = get_encoding(MAX_WORDS, EMB_SIZE).numpy()

        # Assert
        np.testing.assert_array_equal(result, expected_result)
