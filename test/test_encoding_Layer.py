from unittest import TestCase
import numpy as np
from layer_encoder.encoder import EncodingLayer


class TestEncoding_Layer(TestCase):
    def test_compute_attention_mask(self):

        # Arrange
        BATCH_SIZE = 32
        N_CONTEXT = 400
        W_VOCAB_SIZE = 10000

        encoding_layer = EncodingLayer(
            d_model=128,
            kernel_size=7,
            n_conv_layers=4,
            n_heads=8,
            survival_prob=1.0,
            block_num=1,
            dropout_rate=0.1,
            l2_rate=3e-7,

        )
        w_context = np.random.randint(1, W_VOCAB_SIZE, (BATCH_SIZE, N_CONTEXT))

        # Force some random padding in the input
        for row in range(w_context.shape[0]):
            n_pad = np.random.randint(0, 16)
            if n_pad > 0:
                w_context[row][-n_pad:] = 0
        context_word_mask = w_context != 0

        expected_result = np.zeros((BATCH_SIZE, N_CONTEXT, N_CONTEXT), dtype=bool)

        for b in range(BATCH_SIZE):
            for row in range(N_CONTEXT):
                if context_word_mask[b, row] == True:
                    expected_result[b, row] = context_word_mask[b]

        # Act
        result = encoding_layer.compute_attention_mask(context_word_mask)

        # Assert
        np.testing.assert_equal(result, expected_result)
