from metrics import F1Score
from model.question_answer_model import QACNNnet
from Config import *


# Build model and compile
def build_model(input_embedding_params, embedding_encoder_params, conv_layer_params, model_encoder_params,
                max_context_words, max_query_words, max_chars, optimizer, loss):
    # Model input tensors
    context_words_input = tf.keras.Input(shape=(max_context_words), name="context words")
    context_characters_input = tf.keras.Input(shape=(max_context_words, max_chars), name="context characters")
    query_words_input = tf.keras.Input(shape=(max_query_words), name="query words")
    query_characters_input = tf.keras.Input(shape=(max_query_words, max_chars), name="query characters")

    inputs = [context_words_input, context_characters_input, query_words_input, query_characters_input]

    # Create the model and force a call
    model = QACNNnet(input_embedding_params, embedding_encoder_params, conv_layer_params, model_encoder_params)
    model(inputs)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[F1Score()],
        run_eagerly=True
    )

    return model


def custom_accuracy(y_true, y_pred):
    """

    :param y_true: expected shape (batch_size, 2, 1)
    :param y_pred: expected shape (batch_size, 2, 400)
    :return:
    """

    assert y_true.shape[1] == 2
    assert y_pred.shape[1] == 2

    # Number of elements in the current batch
    batch_size = y_true.shape[0]
    n_samples = tf.math.multiply(batch_size, 2)
    n_samples = tf.cast(n_samples, tf.dtypes.int64)

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

    # Count the number of elements that are equal
    n_equal_start = tf.math.count_nonzero(tf.math.equal(y_pred_start, y_true_start))
    n_equal_end = tf.math.count_nonzero(tf.math.equal(y_pred_end, y_true_end))

    return tf.math.divide(tf.math.add(n_equal_start, n_equal_end), n_samples)


# TODO: keep just one version of the loss
def loss_function(y_true, y_pred):
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

    _res3 = tf.reduce_mean(neg_log_sum)

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

    return _res3


def generate_random_data(n_items):
    w_context = np.random.randint(1, WORD_VOCAB_SIZE, (n_items, MAX_CONTEXT_WORDS))
    # Force some random padding in the input
    for row in range(w_context.shape[0]):
        n_pad = np.random.randint(0, 16)
        if n_pad > 0:
            w_context[row][-n_pad:] = 0

    c_context = np.random.randint(0, 100, (n_items, MAX_CONTEXT_WORDS, MAX_CHARS))

    w_query = np.random.randint(1, WORD_VOCAB_SIZE, (n_items, MAX_QUERY_WORDS))
    # Force some random padding in the input
    for row in range(w_query.shape[0]):
        n_pad = np.random.randint(0, 5)
        if n_pad > 0:
            w_query[row][-n_pad:] = 0

    c_query = np.random.randint(0, 100, (n_items, MAX_QUERY_WORDS, MAX_CHARS))

    labels = tf.random.uniform(shape=(n_items, 2, 1), minval=0, maxval=MAX_CONTEXT_WORDS, dtype=tf.int64)

    return w_context, c_context, w_query, c_query, labels


def main():
    model = build_model(input_embedding_params,
                        embedding_encoder_params,
                        conv_layer_params,
                        model_encoder_params,
                        MAX_CONTEXT_WORDS,
                        MAX_QUERY_WORDS,
                        MAX_CHARS,
                        OPTIMIZER,
                        loss_function)

    print("Model succesfully built!")
    model.summary()
    # tf.keras.utils.plot_model(model, "Architecture.png", show_shapes=True, expand_nested=True)

    train_w_context, \
    train_c_context, train_w_query, \
    train_c_query, train_labels = generate_random_data(32)

    valid_w_context, \
    valid_c_context, valid_w_query, \
    valid_c_query, valid_labels = generate_random_data(8)

    '''
    # TEST CODE (will be removed soon)
    labels = tf.random.uniform(shape=(32, 2, 1), minval=0, maxval=MAX_CONTEXT_WORDS, dtype=tf.int64).numpy()

    predictions = tf.random.uniform(shape=(32, 2, 400), dtype=tf.float32).numpy()
    predictions = tf.nn.softmax(predictions).numpy()

    # labels[0][0] = 5
    # predictions[0][0][5] = 0.9999999

    # f1_score(labels, predictions)
    # loss_function(labels, predictions)
    '''

    history = model.fit(
        x=[train_w_context, train_c_context, train_w_query, train_c_query],
        y=train_labels,
        validation_data=(
            [valid_w_context, valid_c_context, valid_w_query, valid_c_query],
            valid_labels),
        verbose=1,
        batch_size=4,
        epochs=5)

    print()


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
    tensor = tf.range(0, context.shape[1])
    tensor_tiled = tf.tile(tensor, [context.shape[0]])
    tensor_tiled_reshaped = tf.reshape(tensor_tiled, [context.shape[0], -1])

    # Create masks to filter out unwanted positions
    mask1 = tensor_tiled_reshaped >= start_indices
    mask2 = tensor_tiled_reshaped <= end_indices
    final_mask = tf.math.logical_and(mask1, mask2)
    final_mask = tf.cast(final_mask, tf.dtypes.int32)

    # Multiply the original token tensor with the mask
    # (unwanted positions will be converted to 0)
    tokens_masked = tf.math.multiply(context, final_mask)

    return tokens_masked


if __name__ == '__main__':
    w_context = tf.constant([
        [1, 2, 3, 1, 5, 4, 5, 0],
        [11, 5, 13, 3, 5, 1, 9, 5],
        [2, 3, 4, 1, 2, 3, 6, 7]
    ])

    # Example of expected y_true (BATCH_SIZE, 2, 1)
    y_true = tf.constant([
        [
            [2], [6], # 2 = start index for FIRST sentence in the batch
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

    # Extract start/end indices
    y_true_start = y_true[:, 0]
    y_true_end = y_true[:, 1]

    # This is the result after the softmax on the predictions
    y_pred_start = tf.constant([[2], [4], [5]])
    y_pred_end = tf.constant([[5], [7], [4]])

    true_tokens = get_answers(w_context, y_true_start, y_true_end)
    pred_tokens = get_answers(w_context, y_pred_start, y_pred_end)

    b_size = w_context.shape[0]
    vocab_size = 10

    tokens_to_ignore = tf.constant([[0], [1], [8]])
    updates = tf.ones(tokens_to_ignore.shape[0])
    full_vocab_tensor = tf.ones(vocab_size)

    tokens_to_ignore_mask = tf.tensor_scatter_nd_sub(full_vocab_tensor, tokens_to_ignore, updates)
    tokens_to_ignore_mask = tf.cast(tokens_to_ignore_mask, tf.dtypes.int32)
    tokens_to_ignore_mask = tf.tile(tokens_to_ignore_mask, [b_size])
    tokens_to_ignore_mask = tf.reshape(tokens_to_ignore_mask, [b_size, -1])

    true_token_bins = tf.math.bincount(true_tokens, minlength=vocab_size, maxlength=vocab_size, axis=-1)
    true_token_bins = tf.math.multiply(true_token_bins, tokens_to_ignore_mask)

    pred_token_bins = tf.math.bincount(pred_tokens, minlength=vocab_size, maxlength=vocab_size, axis=-1)
    pred_token_bins = tf.math.multiply(pred_token_bins, tokens_to_ignore_mask)

    common_token_mask = tf.cast(tf.math.multiply(true_token_bins, pred_token_bins) > 0, tf.dtypes.int32)
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

    f1_score = 2 * (prec * rec) / (prec + rec)
    f1_score = tf.reduce_mean(f1_score)  # Check this...
    print()

    main()
