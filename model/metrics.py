import sys

import tensorflow as tf
from model.inference import get_predictions
from utils import assert_tensor_validity


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, vocab_size=10, ignore_tokens=tf.constant([[0]]), name='f1_score', **kwargs):
        '''
        Create the F1 score metric.

        Parameters:
        -----------
        vocab_size: int
            The vocaboulary size
        ignore_tokens: tf.tensor
            Tensor containing an array of tokens that must be ignored when computing the model metrics
        name: string
            The metric name
        '''

        super(F1Score, self).__init__(name=name, **kwargs)

        # Define variables
        self.w_context = None
        self.f1_score = self.add_weight(name='score', initializer='zeros')
        self.batch_idx = self.add_weight(name='batch_idx', initializer='zeros')
        self.batch_idx.assign_add(1.0)
        self.vocab_size = vocab_size
        self.ignore_tokens = ignore_tokens

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Get the batch size
        b_size = tf.shape(self.w_context)[0]

        # Get indices of the
        y_true_start, y_true_end, y_pred_start, y_pred_end = _split_start_end_indices(y_true, y_pred)

        y_pred_start, y_pred_end = get_predictions(y_pred_start, y_pred_end)

        true_tokens = _get_answers(self.w_context, y_true_start, y_true_end)
        pred_tokens = _get_answers(self.w_context, y_pred_start, y_pred_end)

        tokens_to_ignore_mask = self._get_ignore_tokens_mask(self.ignore_tokens, self.vocab_size, b_size)

        true_token_bins = self._bin_count(true_tokens, tokens_to_ignore_mask, self.vocab_size)
        pred_token_bins = self._bin_count(pred_tokens, tokens_to_ignore_mask, self.vocab_size)

        len_common_tokens = self._count_common_tokens(true_token_bins, pred_token_bins)
        len_true_token = self._count_tokens(true_token_bins)
        len_pred_token = self._count_tokens(pred_token_bins)

        current_f1_score = self._compute_f1_score(len_true_token, len_pred_token, len_common_tokens)
        self.f1_score.assign(((self.f1_score * (self.batch_idx - 1)) + current_f1_score)/self.batch_idx)
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

    def _get_ignore_tokens_mask(self, ignore_tokens, vocab_size, b_size):
        '''
        When computing the metric some tokens (e.g., punctuations) must be ignored.
        The goal of this method is to create a mask that will be used for removing them.
        '''

        updates = tf.ones(ignore_tokens.shape[0], dtype=tf.dtypes.int64)
        full_vocab_tensor = tf.ones(vocab_size, dtype=tf.dtypes.int64)

        tokens_to_ignore_mask = tf.tensor_scatter_nd_sub(full_vocab_tensor, ignore_tokens, updates)
        tokens_to_ignore_mask = tf.cast(tokens_to_ignore_mask, tf.dtypes.int64)
        tokens_to_ignore_mask = tf.tile(tokens_to_ignore_mask, [b_size])
        tokens_to_ignore_mask = tf.reshape(tokens_to_ignore_mask, [b_size, -1])

        return tokens_to_ignore_mask

    def _bin_count(self, tokens, tokens_to_ignore_mask, vocab_size):
        '''
        Utility method used for counting the number of occurences of each possible token that
        is either in the ground truth or in the predictions
        '''

        token_bins = tf.math.bincount(tokens, minlength=vocab_size, maxlength=vocab_size, axis=-1,
                                      dtype=tf.dtypes.int64)
        token_bins = tf.math.multiply(token_bins, tokens_to_ignore_mask)

        return token_bins

    def _count_common_tokens(self, true_token_bins, pred_token_bins):
        '''
        Utility method for counting the number of common tokens between the ground truth bins
        and the predictions bins
        '''

        common_token_mask = tf.cast(tf.math.multiply(true_token_bins, pred_token_bins) > 0, tf.dtypes.int64)
        len_common_tokens = tf.math.minimum(tf.math.multiply(true_token_bins, common_token_mask),
                                            tf.math.multiply(pred_token_bins, common_token_mask))
        len_common_tokens = tf.math.reduce_sum(len_common_tokens, axis=-1)
        return len_common_tokens

    def _count_tokens(self, token_bins):
        '''
        Utility method for counting the total number of token that are within a bin
        '''

        len_token = tf.math.reduce_sum(token_bins, axis=-1)
        return len_token

    def _compute_f1_score(self, len_true_token, len_pred_token, len_common_tokens):
        '''
        Effectively compute the F1-SCORE. (Refer to the relation for further detail)
        '''

        epsilon = 1e-6
        len_true_token = tf.cast(len_true_token, tf.float32)
        len_pred_token = tf.cast(len_pred_token, tf.float32)
        len_common_tokens = tf.cast(len_common_tokens, tf.float32)

        prec = (len_common_tokens / (len_pred_token + epsilon)) + epsilon
        rec = (len_common_tokens / (len_true_token + epsilon)) + epsilon

        '''
        tf.print('len_common_tokens: ', len_common_tokens)
        tf.print('len_pred_token: ', len_pred_token)
        tf.print('len_true_token: ', len_true_token)
        '''

        f1_score_values = 2 * (prec * rec) / (prec + rec)

        return tf.reduce_mean(f1_score_values)


class EMScore(tf.keras.metrics.Metric):

    def __init__(self, vocab_size=10, ignore_tokens=tf.constant([[0]]), name='em_score', **kwargs):
        '''
        Create the EM score metric

        Parameters:
        -----------
        vocab_size: int
            The vocaboulary size
        ignore_tokens: tf.tensor
            Tensor containing an array of tokens that must be ignored when computing the model metrics
        name: string
            The metric name
        '''

        super(EMScore, self).__init__(name=name, **kwargs)

        # Define variables
        self.w_context = None
        self.em_score = self.add_weight(name='score', initializer='zeros')
        self.batch_idx = self.add_weight(name='batch_idx', initializer='zeros')
        self.batch_idx.assign_add(1.0)
        self.vocab_size = vocab_size
        self.ignore_tokens = ignore_tokens

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Split true and predictions to single tensors
        y_true_start, y_true_end, y_pred_start, y_pred_end = _split_start_end_indices(y_true, y_pred)

        # Get the predictions leveraging the "inference algorithm" (refer to the relation for details)
        y_pred_start, y_pred_end = get_predictions(y_pred_start, y_pred_end)

        # Extract the answers from the dataset, based on true and predicted indices
        true_tokens = _get_answers(self.w_context, y_true_start, y_true_end)
        pred_tokens = _get_answers(self.w_context, y_pred_start, y_pred_end)

        # Get a mask for discarding 0-valued true tokens
        true_tokens_mask = tf.math.not_equal(true_tokens, 0)

        # Count the number of common tokens
        common_tokens = tf.math.logical_and(
            tf.math.equal(true_tokens, pred_tokens),
            true_tokens_mask
        )
        common_tokens = tf.cast(common_tokens, tf.dtypes.float32)
        common_tokens = tf.reduce_sum(common_tokens)

        # Count the number of total true tokens
        epsilon = 1e-6
        true_tokens = tf.reduce_sum(tf.cast(true_tokens_mask, tf.dtypes.float32))

        # Compute the em_score
        current_em_score = common_tokens / (true_tokens + epsilon)

        # Update the metric
        self.em_score.assign(((self.em_score * (self.batch_idx - 1)) + current_em_score) / self.batch_idx)
        self.batch_idx.assign_add(1.0)

        # Reset variables
        self.w_context = None

    def set_words_context(self, words):
        # This method will be called after each mini-batch, because the tensor containing
        # the contexts will change at each iteration.
        self.w_context = words

    def result(self):
        return self.em_score

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.em_score.assign(0.0)
        self.batch_idx.assign(1.0)


def _get_answers(context, start_indices, end_indices):
    '''
    Create a new tensor that contains slice of the original context

    Parameters:
    -----------
    context: tf.tensor
        the original context words passed as input to the network
    start_indices: tf.tensor
        array that contains the predicted start indices
    end_indices: tf.tensor
        array that contains the predicted end indices

    Returns:
    --------
    tokens_masked: tf.tensor
        input token tensor appropriately masked
    '''

    # Check dimensions
    assert (context.shape[0] == start_indices.shape[0])
    assert (context.shape[0] == end_indices.shape[0])

    b_size = tf.shape(context)[0]

    # Create a tensor that has the same token shape, and
    # that contains just position indices
    tensor = tf.range(0, context.shape[1], dtype=tf.dtypes.int64)
    tensor_tiled = tf.tile(tensor, [b_size])
    tensor_tiled_reshaped = tf.reshape(tensor_tiled, [b_size, -1])

    # Create masks to filter out unwanted positions
    mask1 = tensor_tiled_reshaped >= start_indices
    mask2 = tensor_tiled_reshaped <= end_indices
    final_mask = tf.math.logical_and(mask1, mask2)
    final_mask = tf.cast(final_mask, tf.dtypes.int64)

    context = tf.cast(context, tf.dtypes.int64)

    # Multiply the original token tensor with the mask
    # (unwanted positions will be converted to 0)
    tokens_masked = tf.math.multiply(context, final_mask)

    return tokens_masked


def qa_loss(y_true, y_pred):
    '''
    Compute the loss that must be minimized during the model training

    Parameters:
    -----------
    y_true: tf.tensor
        Tensor contaning the ground truth. Valid shapes are (batch_size, 2, 1) or (batch_size, 2)
    y_pred: tf.tensor
        Tensor containing the predictions. Valid shape is (batch_size, 2, n_words)
    '''

    epsilon = 1e-6

    # Clip values for numerical stability
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    assert y_true.shape[1] == 2
    assert y_pred.shape[1] == 2

    y_true_start, y_true_end = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

    # assert_tensor_validity(y_true, "y_true")
    assert_tensor_validity(y_pred, "y_pred")  # This contains nan at a certain point
    # assert_tensor_validity(y_true_start, "y_true_start")
    # assert_tensor_validity(y_true_end, "y_true_end")

    # Get the probabilities of the corresponding ground truth
    p1 = tf.gather(params=y_pred_start, indices=y_true_start, axis=-1, batch_dims=-1)
    p2 = tf.gather(params=y_pred_end, indices=y_true_end, axis=-1, batch_dims=-1)

    # Reshape probabilities by removing unnecessary dimensions
    p1 = tf.reshape(p1, shape=(-1, 1))
    p2 = tf.reshape(p2, shape=(-1, 1))

    # Compute the loss
    log_p1 = tf.math.log(p1 + epsilon)
    log_p2 = tf.math.log(p2 + epsilon)

    neg_log_p1 = tf.math.negative(log_p1)
    neg_log_p2 = tf.math.negative(log_p2)

    neg_log_sum = neg_log_p1 + neg_log_p2

    loss = tf.reduce_mean(neg_log_sum)

    assert_tensor_validity(p1, "p1")  # This contains nan at a certain point
    assert_tensor_validity(p2, "p2")
    assert_tensor_validity(log_p1, "log_p1")
    assert_tensor_validity(log_p2, "log_p2")
    assert_tensor_validity(neg_log_p1, "neg_log_p1")
    assert_tensor_validity(neg_log_p2, "neg_log_p2")
    assert_tensor_validity(neg_log_sum, "neg_log_sum")

    '''
    tf.print("p1:", p1, output_stream=sys.stdout)
    tf.print("p2:", p2, output_stream=sys.stdout)
    tf.print("log_p1:", log_p1, output_stream=sys.stdout)
    tf.print("log_p2:", log_p2, output_stream=sys.stdout)
    tf.print("neg_log_p1:", neg_log_p1, output_stream=sys.stdout)
    tf.print("neg_log_p2:", neg_log_p2, output_stream=sys.stdout)
    tf.print("neg_log_sum:", neg_log_sum, output_stream=sys.stdout)
    tf.print("loss:", loss, output_stream=sys.stdout)
    tf.print("****************************************")
    '''

    return loss


def _split_start_end_indices(y_true, y_pred):
    '''
    Create tensors containing true and predicted indices referring to
    both start and end answer positions.

    Parameters:
    -----------
    y_true: tf.tensor
        Tensor contaning the ground truth. Valid shapes are (batch_size, 2, 1) or (batch_size, 2)
    y_pred: tf.tensor
        Tensor containing the predictions. Valid shape is (batch_size, 2, n_words)
    '''

    y_true_start, y_true_end = y_true[:, 0], y_true[:, 1]
    y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

    return y_true_start, y_true_end, y_pred_start, y_pred_end
