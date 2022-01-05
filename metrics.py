import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.words_context = None
        self.f1_score = 0

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Split the data
        y_true_start, y_true_end = tf.split(y_true, num_or_size_splits=2, axis=1)
        y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

        # Remove unused dimension from labels
        y_true_start = tf.squeeze(y_true_start, axis=1)
        y_true_start = tf.cast(y_true_start, tf.dtypes.int64)

        y_true_end = tf.squeeze(y_true_end, axis=1)
        y_true_end = tf.cast(y_true_end, tf.dtypes.int64)

        # Get the predictions
        y_pred_start = tf.argmax(y_pred_start, axis=-1, output_type=tf.dtypes.int64)
        y_pred_end = tf.argmax(y_pred_end, axis=-1, output_type=tf.dtypes.int64)

        tf.print(y_pred_start, y_pred_end)

        '''
        truth_tokens = self.words_context[y_true[0]:y_true[0]]
        pred_tokens = self.words_context[y_true[0]:y_true[0]]  # TODO: update

        common_tokens = tf.sets.intersection(
            truth_tokens, pred_tokens, validate_indices=True
        ).values

        # Values to filter
        filter_values = tf.constant([[0], [1]])
        # Number of values to filter
        n_values_to_filter = filter_values.shape[0]

        # Tile the tensor that must be filtered
        filter_values_tiled = tf.tile(common_tokens, [n_values_to_filter])
        filter_values_tiled = tf.reshape(filter_values_tiled, [n_values_to_filter, -1])
        # Create the mask
        bool_mask = tf.not_equal(filter_values_tiled, filter_values)
        # Reduce according to x axis
        bool_mask = tf.math.reduce_all(bool_mask, 0)
        common_tokens = tf.boolean_mask(common_tokens, bool_mask)

        truth_tokens = tf.reshape(truth_tokens, [1, -1])

        bool_mask = tf.not_equal(truth_tokens, 0)
        truth_tokens = tf.boolean_mask(truth_tokens, bool_mask)

        pred_tokens = tf.reshape(pred_tokens, [1, -1])
        bool_mask = tf.not_equal(pred_tokens, 0)
        pred_tokens = tf.boolean_mask(pred_tokens, bool_mask)

        prec = common_tokens.shape[0] / pred_tokens.shape[0]
        rec = common_tokens.shape[0] / truth_tokens.shape[0]

        f1_score = 2 * (prec * rec) / (prec + rec)



        
        '''
        # Reset variables
        self.words_context = None

    def set_words_context(self, words):
        self.words_context = words

    def result(self):
        return self.f1_score

