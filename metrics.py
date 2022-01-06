import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, vocab_size=10, ignore_tokens=tf.constant([[0]]), name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.w_context = None
        self.f1_score = self.add_weight(name='score', initializer='zeros')
        self.batch_idx = self.add_weight(name='batch_idx', initializer='zeros')
        self.batch_idx.assign_add(1.0)
        self.vocab_size = vocab_size
        self.ignore_tokens = ignore_tokens

    def update_state(self, y_true, y_pred, sample_weight=None):

        b_size = self.w_context.shape[0]

        # Extract start/end indices
        y_true_start, y_true_end = y_true[:, 0], y_true[:, 1]
        y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

        y_pred_start = tf.argmax(y_pred_start, axis=-1, output_type=tf.dtypes.int64)
        y_pred_end = tf.argmax(y_pred_end, axis=-1, output_type=tf.dtypes.int64)

        true_tokens = get_answers(self.w_context, y_true_start, y_true_end)
        pred_tokens = get_answers(self.w_context, y_pred_start, y_pred_end)

        # tokens_to_ignore = tf.constant([[0], [1], [8]])
        updates = tf.ones(self.ignore_tokens.shape[0], dtype=tf.dtypes.int64)
        full_vocab_tensor = tf.ones(self.vocab_size, dtype=tf.dtypes.int64)

        tokens_to_ignore_mask = tf.tensor_scatter_nd_sub(full_vocab_tensor, self.ignore_tokens, updates)
        tokens_to_ignore_mask = tf.cast(tokens_to_ignore_mask, tf.dtypes.int64)
        tokens_to_ignore_mask = tf.tile(tokens_to_ignore_mask, [b_size])
        tokens_to_ignore_mask = tf.reshape(tokens_to_ignore_mask, [b_size, -1])

        true_token_bins = tf.math.bincount(true_tokens, minlength=self.vocab_size, maxlength=self.vocab_size, axis=-1, dtype=tf.dtypes.int64)
        true_token_bins = tf.math.multiply(true_token_bins, tokens_to_ignore_mask)

        pred_token_bins = tf.math.bincount(pred_tokens, minlength=self.vocab_size, maxlength=self.vocab_size, axis=-1, dtype=tf.dtypes.int64)
        pred_token_bins = tf.math.multiply(pred_token_bins, tokens_to_ignore_mask)

        common_token_mask = tf.cast(tf.math.multiply(true_token_bins, pred_token_bins) > 0, tf.dtypes.int64)
        len_common_tokens = tf.math.minimum(tf.math.multiply(true_token_bins, common_token_mask),
                                            tf.math.multiply(pred_token_bins, common_token_mask))
        len_common_tokens = tf.math.reduce_sum(len_common_tokens, axis=-1)

        len_true_token = tf.math.reduce_sum(true_token_bins, axis=-1)
        len_pred_token = tf.math.reduce_sum(pred_token_bins, axis=-1)

        # Avoid divisions by 0
        epsilon = 1e-8
        len_true_token = tf.cast(len_true_token, tf.float32)
        len_pred_token = tf.cast(len_pred_token, tf.float32)
        len_common_tokens = tf.cast(len_common_tokens, tf.float32)

        prec = (len_common_tokens / (len_pred_token + epsilon)) + epsilon
        rec = (len_common_tokens / (len_true_token + epsilon)) + epsilon

        f1_score_values = 2 * (prec * rec) / (prec + rec)

        self.f1_score.assign_add((self.f1_score + tf.reduce_mean(f1_score_values))/self.batch_idx)
        self.batch_idx.assign_add(1.0)
        # Reset variables
        self.w_context = None

    def set_words_context(self, words):
        self.w_context = words

    def result(self):
        return self.f1_score

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.f1_score.assign(0.0)
        self.batch_idx.assign(1.0)


def get_answers(context, start_indices, end_indices):
    """
    Create a new tensor that contains slice of the original context
    :param context: the original context words passed as input to the network
    :param start_indices: array that contains the predicted start indices
    :param end_indices: array that contains the predicted end indices
    :return: tokens_masked: input token tensor appropriately masked
    """
    # Check dimensions
    assert (context.shape[0] == start_indices.shape[0])
    assert (context.shape[0] == end_indices.shape[0])

    # Create a tensor that has the same token shape, and
    # that contains just position indices
    tensor = tf.range(0, context.shape[1], dtype=tf.dtypes.int64)
    tensor_tiled = tf.tile(tensor, [context.shape[0]])
    tensor_tiled_reshaped = tf.reshape(tensor_tiled, [context.shape[0], -1])

    # Create masks to filter out unwanted positions
    mask1 = tensor_tiled_reshaped >= start_indices
    mask2 = tensor_tiled_reshaped <= end_indices
    final_mask = tf.math.logical_and(mask1, mask2)
    final_mask = tf.cast(final_mask, tf.dtypes.int64)

    # Multiply the original token tensor with the mask
    # (unwanted positions will be converted to 0)
    tokens_masked = tf.math.multiply(context, final_mask)

    return tokens_masked


def qa_loss(y_true, y_pred):
    # y_true = (batch_size, 2, 1) or (batch_size, 2)
    # y_pred = (batch_size, 2, n_words)
    epsilon = 1e-8

    assert y_true.shape[1] == 2
    assert y_pred.shape[1] == 2

    batch_size = y_true.shape[0]

    y_true_start, y_true_end = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

    # Get the probabilities of the corresponding ground truth
    p1 = tf.gather(params=y_pred_start, indices=y_true_start, axis=-1, batch_dims=-1)
    p2 = tf.gather(params=y_pred_end, indices=y_true_end, axis=-1, batch_dims=-1)

    p1 = tf.reshape(p1, shape=(batch_size, 1))
    p2 = tf.reshape(p2, shape=(batch_size, 1))

    log_p1 = tf.math.log(p1 + epsilon)
    log_p2 = tf.math.log(p2 + epsilon)

    neg_log_p1 = tf.math.negative(log_p1)
    neg_log_p2 = tf.math.negative(log_p2)

    neg_log_sum = neg_log_p1 + neg_log_p2

    loss = tf.reduce_mean(neg_log_sum)

    '''
    # Remove unused dimension from labels
    y_true_start = tf.squeeze(y_true_start, axis=1)
    y_true_start = tf.cast(y_true_start, tf.dtypes.int64)

    y_true_end = tf.squeeze(y_true_end, axis=1)
    y_true_end = tf.cast(y_true_end, tf.dtypes.int64)

    # Remove unused dimension from predictions
    y_pred_start = tf.squeeze(y_pred_start, axis=1)
    y_pred_end = tf.squeeze(y_pred_end, axis=1)

    # Create one hot encoding labels
    y_true_start_one_hot = tf.one_hot(y_true_start, 400)
    y_true_start_one_hot = tf.squeeze(y_true_start_one_hot, axis=1)

    y_true_end_one_hot = tf.one_hot(y_true_end, 400)
    y_true_end_one_hot = tf.squeeze(y_true_end_one_hot, axis=1)

    # EXAMPLE WITH MANUAL COMPUTATION OF THE LOSS
    a = -tf.reduce_sum(y_true_start_one_hot * tf.math.log(y_pred_start + 1e-8)) / batch_size
    b = -tf.reduce_sum(y_true_end_one_hot * tf.math.log(y_pred_end + 1e-8)) / batch_size
    _res0 = tf.reduce_mean(a + b)

    # EXAMPLE WITH SPARSE CATEGORICAL CROSS ENTROPY
    loss1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    a = loss1(y_true_start, y_pred_start)
    b = loss1(y_true_end, y_pred_end)
    _res1 = tf.reduce_mean(a + b)

    # EXAMPLE WITH CATEGORICAL CROSS ENTROPY
    loss2 = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    a2 = loss2(y_true_start_one_hot, y_pred_start)
    b2 = loss2(y_true_end_one_hot, y_pred_end)
    _res2 = tf.reduce_mean(a2 + b2)
    
    tf.print('Manual: ', _res0)
    tf.print('SparseCategoricalCE: ', _res1)
    tf.print('CategoricalCE:', _res2)
    tf.print('Paper: ', _res3)
    '''

    return loss
