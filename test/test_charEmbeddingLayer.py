from unittest import TestCase

import tensorflow as tf
import numpy as np

from layer_input_embedding.char_embedding_layer import CharEmbeddingLayer


class TestCharEmbeddingLayer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.BATCH_SIZE = 2
        cls.N_WORDS = 2
        cls.N_CHAR = 6
        cls.VOCAB_SIZE = 50
        cls.EMB_SIZE = 5
        cls.CONV_OUTPUT_SIZE = 5
        cls.CONV_KERNEL_SIZE = 3

        cls.sentence_input = np.random.randint(
            cls.VOCAB_SIZE, size=(cls.BATCH_SIZE, cls.N_WORDS, cls.N_CHAR)
        )

    def test_call_checkOutputShapeValidity(self):

        # Arrange
        char_emb_layer = CharEmbeddingLayer(
            emb_size=self.EMB_SIZE,
            vocab_size=self.VOCAB_SIZE,
            conv_kernel_size=self.CONV_KERNEL_SIZE,
            conv_output_size=self.CONV_OUTPUT_SIZE
            )

        # Act
        result = char_emb_layer(self.sentence_input)

        # Assert
        self.assertEqual(result.shape, (self.BATCH_SIZE, self.N_WORDS, self.CONV_OUTPUT_SIZE))

    def test_call_checkMaxpoolValidity(self):

        # Arrange
        char_emb_layer = CharEmbeddingLayer(
            emb_size=self.EMB_SIZE,
            vocab_size=self.VOCAB_SIZE,
            conv_kernel_size=self.CONV_KERNEL_SIZE,
            conv_output_size=self.CONV_OUTPUT_SIZE
            )

        # Act
        fake_conv_output = np.random.randint(
            100, size=(self.BATCH_SIZE, self.N_WORDS, self.N_CHAR, self.CONV_OUTPUT_SIZE)
        )

        expected_output = np.zeros((self.BATCH_SIZE, self.N_WORDS, self.CONV_OUTPUT_SIZE))
        for i, sentence in enumerate(fake_conv_output):
            for j, word in enumerate(sentence):
                for filter_idx in range(word.shape[1]):
                    expected_output[i, j, filter_idx] = np.max(word[:, filter_idx])

        result = char_emb_layer._maxpool(fake_conv_output)

        # Assert
        np.testing.assert_array_equal(expected_output, result)
