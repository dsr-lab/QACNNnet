from unittest import TestCase

import numpy as np
import tensorflow as tf
from word_embedding_layer import WordEmbeddingLayer


class TestWordEmbeddingLayer(TestCase):
    def test_build(self):
        self.fail()

    def test_call(self):
        # Arrange
        emb_size = 5
        emb_matrix = np.asarray([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20]  # <UNK>
        ])
        # 4 words
        # 0 = padding
        # 1 - 19 = vocab_size
        # 20 = <UNK>
        n_special_words = 1
        vocab_size = len(emb_matrix)  # - n_special_words + 1  # +1 for padding

        s = WordEmbeddingLayer(emb_size, emb_matrix, vocab_size, n_special_words)

        sentence_input = np.asarray(
            [[1, 2, 3, 2, 4, 3, 0, 0]]
        )


        # Act
        res = s(sentence_input)

        # Assert

        self.fail()

    def test__set_weights(self):
        self.fail()
