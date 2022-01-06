from unittest import TestCase
import tensorflow as tf

class TestF1Score(TestCase):

    def test_update_state(self):
        w_context = tf.constant([
            [1, 2, 3, 1, 5, 4, 5, 0],
            [11, 5, 13, 3, 5, 1, 9, 5],
            [2, 3, 4, 1, 2, 3, 6, 7]
        ])

        # Example of expected y_true (BATCH_SIZE, 2, 1)
        y_true = tf.constant([
            [
                [2], [6],  # 2 = start index for FIRST sentence in the batch
                # 6 = end index for the FIRST sentence in the batch
            ],
            [
                [0], [5]  # 0 = start index for SECOND sentence in the batch
                # 5 = end index for the SECOND sentence in the batch
            ],
            [
                [3], [5]
            ]
        ])

        y_pred = tf.constant([
            [
                [2,2,3,4,5,6], [6,2,3,4,5,6],  # 2 = start index for FIRST sentence in the batch
                # 6 = end index for the FIRST sentence in the batch
            ],
            [
                [0,2,3,4,5,6], [5,2,3,4,5,6]  # 0 = start index for SECOND sentence in the batch
                # 5 = end index for the SECOND sentence in the batch
            ],
            [
                [3,2,3,4,5,6], [5,2,3,4,5,6]
            ]
        ])

        #y_pred_start = y_pred[:, 0]
        #y_pred_end = y_pred[:, 1]

        y_true_start, y_true_end = y_true[:, 0], y_true[:, 1]
        y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

        y_pred_start = tf.argmax(y_pred_start, axis=-1, output_type=tf.dtypes.int64)
        y_pred_end = tf.argmax(y_pred_end, axis=-1, output_type=tf.dtypes.int64)


        # This is the result after the softmax on the predictions
        y_pred_start = tf.constant([[2], [4], [5]])
        y_pred_end = tf.constant([[5], [7], [4]])

        self.fail()
