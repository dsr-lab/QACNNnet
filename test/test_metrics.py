from unittest import TestCase
import numpy as np

from model.metrics import *
from model.metrics import _get_answers, _split_start_end_indices


class TestMetrics(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.w_context = tf.constant([
            [1, 2, 3, 1, 5, 4, 5, 0],
            [11, 5, 13, 3, 5, 1, 9, 5],
            [2, 3, 4, 1, 2, 3, 6, 7]
        ], dtype=tf.dtypes.int64)

        # Example of expected y_true (BATCH_SIZE, 2, 1)
        cls.y_true = tf.constant([
            [
                [2], [6],   # 2 = start index for FIRST sentence in the batch
                            # 6 = end index for the FIRST sentence in the batch
            ],
            [
                [0], [5]    # 0 = start index for SECOND sentence in the batch
                            # 5 = end index for the SECOND sentence in the batch
            ],
            [
                [3], [5]
            ]
        ], dtype=tf.dtypes.int64)

        cls.y_pred = tf.constant([
            [
                [0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            [
                [0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ],
            [
                [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            ]
        ])
        cls.vocab_size = 15
        cls.ignore_tokens = tf.constant([[0], [1], [8]])

    def test_get_answers(self):
        # Arrange
        f1_score = F1Score(vocab_size=self.vocab_size, ignore_tokens=self.ignore_tokens)

        # Act
        y_true_start, y_true_end, y_pred_start, y_pred_end = _split_start_end_indices(self.y_true, self.y_pred)
        y_pred_start, y_pred_end = get_predictions(y_pred_start, y_pred_end, n_words=self.y_pred.shape[-1])

        true_tokens = _get_answers(self.w_context, y_true_start, y_true_end)
        pred_tokens = _get_answers(self.w_context, y_pred_start, y_pred_end)

        expected_true_tokens = tf.constant([
            [0, 0, 3, 1, 5, 4, 5, 0],
            [11, 5, 13, 3, 5, 1, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0]
        ])

        expected_pred_tokens = tf.constant([
            [0, 0, 0, 1, 5, 4, 5, 0],
            [0, 5, 13, 3, 5, 0, 0, 0],
            [2, 3, 4, 1, 2, 3, 0, 0]
        ])

        # Assert
        np.testing.assert_array_equal(true_tokens, expected_true_tokens)
        np.testing.assert_array_equal(pred_tokens, expected_pred_tokens)

    def test_get_tokens_to_ignore_mask(self):
        # Arrange
        f1_score = F1Score(vocab_size=self.vocab_size, ignore_tokens=self.ignore_tokens)
        f1_score.set_words_context(self.w_context)

        b_size = self.w_context.shape[0]

        # Act
        tokens_to_ignore_mask = f1_score._get_ignore_tokens_mask(self.ignore_tokens, self.vocab_size, b_size)
        expected_tokens_to_ignore_mask = tf.constant(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
            ]
        )

        # Assert
        np.testing.assert_array_equal(tokens_to_ignore_mask, expected_tokens_to_ignore_mask)

    def test_bin_count(self):
        # Arrange
        f1_score = F1Score(vocab_size=self.vocab_size, ignore_tokens=self.ignore_tokens)
        f1_score.set_words_context(self.w_context)

        b_size = self.w_context.shape[0]
        y_true_start, y_true_end, y_pred_start, y_pred_end = _split_start_end_indices(self.y_true, self.y_pred)

        y_pred_start, y_pred_end = get_predictions(y_pred_start, y_pred_end, n_words=self.y_pred.shape[-1])

        true_tokens = _get_answers(self.w_context, y_true_start, y_true_end)
        pred_tokens = _get_answers(self.w_context, y_pred_start, y_pred_end)

        tokens_to_ignore_mask = f1_score._get_ignore_tokens_mask(self.ignore_tokens, self.vocab_size, b_size)

        # Act
        true_token_bins = f1_score._bin_count(true_tokens, tokens_to_ignore_mask, self.vocab_size)
        pred_token_bins = f1_score._bin_count(pred_tokens, tokens_to_ignore_mask, self.vocab_size)

        expected_true_token_bins = tf.constant(
            [
                [0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        expected_pred_token_bins = tf.constant(
            [
                [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        # Assert
        np.testing.assert_array_equal(true_token_bins, expected_true_token_bins)
        np.testing.assert_array_equal(pred_token_bins, expected_pred_token_bins)

    def test_count_common_tokens(self):
        # Assert
        b_size = self.w_context.shape[0]
        f1_score = F1Score(vocab_size=self.vocab_size, ignore_tokens=self.ignore_tokens)

        y_true_start, y_true_end, y_pred_start, y_pred_end = _split_start_end_indices(self.y_true, self.y_pred)

        y_pred_start, y_pred_end = get_predictions(y_pred_start, y_pred_end, n_words=self.y_pred.shape[-1])

        true_tokens = _get_answers(self.w_context, y_true_start, y_true_end)
        pred_tokens = _get_answers(self.w_context, y_pred_start, y_pred_end)

        tokens_to_ignore_mask = f1_score._get_ignore_tokens_mask(self.ignore_tokens, self.vocab_size, b_size)

        true_token_bins = f1_score._bin_count(true_tokens, tokens_to_ignore_mask, self.vocab_size)
        pred_token_bins = f1_score._bin_count(pred_tokens, tokens_to_ignore_mask, self.vocab_size)

        # Act
        len_common_tokens = f1_score._count_common_tokens(true_token_bins, pred_token_bins)

        # Assert
        np.testing.assert_array_equal(len_common_tokens, [3, 4, 2])

    def test_count_tokens(self):
        # Arrange
        b_size = self.w_context.shape[0]
        f1_score = F1Score(vocab_size=self.vocab_size, ignore_tokens=self.ignore_tokens)

        y_true_start, y_true_end, y_pred_start, y_pred_end = _split_start_end_indices(self.y_true, self.y_pred)

        y_pred_start, y_pred_end = get_predictions(y_pred_start, y_pred_end, n_words=self.y_pred.shape[-1])

        true_tokens = _get_answers(self.w_context, y_true_start, y_true_end)
        pred_tokens = _get_answers(self.w_context, y_pred_start, y_pred_end)

        tokens_to_ignore_mask = f1_score._get_ignore_tokens_mask(self.ignore_tokens, self.vocab_size, b_size)

        true_token_bins = f1_score._bin_count(true_tokens, tokens_to_ignore_mask, self.vocab_size)
        pred_token_bins = f1_score._bin_count(pred_tokens, tokens_to_ignore_mask, self.vocab_size)

        # Act
        len_true_token = f1_score._count_tokens(true_token_bins)
        len_pred_token = f1_score._count_tokens(pred_token_bins)

        # Assert
        np.testing.assert_array_equal(len_true_token, [4, 5, 2])
        np.testing.assert_array_equal(len_pred_token, [3, 4, 5])

    def test_update_f1_score(self):
        # Arrange
        current_f1_scores = np.array([0.3, 0.2, 0, 0.5, 0.1])
        f1_score = F1Score(vocab_size=self.vocab_size, ignore_tokens=self.ignore_tokens)

        # Act
        for score in current_f1_scores:
            prev_f1_score = f1_score.result()
            prev_batch_idx = f1_score.batch_idx - 1

            current_f1_score = tf.constant(score, dtype=tf.dtypes.float16)
            current_batch_idx = f1_score.batch_idx

            new_f1_score = ((prev_f1_score * prev_batch_idx) + current_f1_score) / f1_score.batch_idx
            f1_score.batch_idx.assign_add(1.0)

            f1_score.f1_score.assign(new_f1_score)

        result = f1_score.result()
        expected_result = tf.constant(np.mean(current_f1_scores), dtype=tf.dtypes.float16)

        # Assert
        np.testing.assert_equal(result, expected_result)
