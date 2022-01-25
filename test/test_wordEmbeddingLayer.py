from unittest import TestCase

import numpy as np
from layer_input_embedding.word_embedding_layer import WordEmbeddingLayer


class TestWordEmbeddingLayer(TestCase):

    def test_call(self):
        # ####################
        # Arrange
        # ####################
        emb_size = 5
        vocab_size = 4
        max_input_length = 8

        # Force float32 precision, otherwise the comparison with tensors would not work correctly.
        emb_matrix = np.random.rand(vocab_size, emb_size).astype(np.float32)

        # 0 is padding reserved, and it is considered in the layer itself
        n_special_words = 1  # just the <UNK>
        vocab_size = len(emb_matrix)

        emb_layer = WordEmbeddingLayer(emb_size, emb_matrix, vocab_size, n_special_words)

        sentences_input = np.asarray(
            [
                [1, 2, 3, 2, 4, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0, 0, 0, 0]
            ]
        )

        # Create the expected output result
        expexted_result = np.zeros((len(sentences_input), max_input_length, emb_size), dtype=np.float32)
        for i, sentence in enumerate(sentences_input):
            for j, token in enumerate(sentence):
                if token != 0:
                    expexted_result[i, j] = emb_matrix[token - 1]

        # ####################
        # Act
        # ####################
        result, _ = emb_layer(sentences_input)

        # ####################
        # Assert
        # ####################
        np.testing.assert_array_equal(result, expexted_result)

