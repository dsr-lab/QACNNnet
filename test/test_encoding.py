from unittest import TestCase
import numpy as np

from layer_encoder.positional_encoding import get_encoding


class TestEncoding(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.MAX_WORDS = 5
        cls.EMB_SIZE = 10
        cls.BATCH_SIZE = 2

    def test_sum_positionalEncoding(self):

        embedding = np.random.rand(self.BATCH_SIZE, self.MAX_WORDS, self.EMB_SIZE)
        pos_encoding = get_encoding(self.MAX_WORDS, self.EMB_SIZE)

        expected_result = np.zeros((self.BATCH_SIZE, self.MAX_WORDS, self.EMB_SIZE))

        for b in range(self.BATCH_SIZE):
            expected_result[b] = embedding[b] + pos_encoding[0]

        result = embedding
        result += pos_encoding

        np.testing.assert_array_equal(result, expected_result)

    def test_call_getEncoding(self):

        # Arrange
        expected_result = np.zeros((1, self.MAX_WORDS, self.EMB_SIZE), dtype=np.float16)
        for b in range(1):
            for i in range(self.MAX_WORDS):
                for j in range(self.EMB_SIZE):
                    if j % 2 == 0:
                        expected_result[b, i, j] = np.sin(i / np.power(10000, 2 * (j//2) / self.EMB_SIZE))
                    else:
                        expected_result[b, i, j] = np.cos(i / np.power(10000, 2 * (j//2) / self.EMB_SIZE))

        # Act
        result = get_encoding(self.MAX_WORDS, self.EMB_SIZE).numpy()

        # Assert
        np.testing.assert_array_equal(result, expected_result)
