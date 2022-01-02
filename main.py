from model.question_answer_model import QACNNnet
from Config import *


# Build model and compile
def build_model(input_embedding_params, embedding_encoder_params, conv_layer_params, model_encoder_params,
                max_context_words, max_query_words, max_chars, optimizer, loss):

    net = QACNNnet(input_embedding_params, embedding_encoder_params, conv_layer_params, model_encoder_params)

    model = net.model(max_context_words, max_query_words, max_chars)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    return model


def custom_accuracy(y_true, y_pred):
    assert y_true.shape[1] == 2
    assert y_pred.shape[1] == 2

    batch_size = y_true.shape[0]

    y_true_start, y_true_end = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

    p1 = tf.gather(params=y_pred_start, indices=y_true_start, axis=-1, batch_dims=-1)
    p2 = tf.gather(params=y_pred_end, indices=y_true_end, axis=-1, batch_dims=-1)

    p1 = tf.reshape(p1, shape=(batch_size, 1))
    p2 = tf.reshape(p2, shape=(batch_size, 1))




def loss_function(y_true, y_pred):
    # y_true = (batch_size, 2, 1) or (batch_size, 2)
    # y_pred = (batch_size, 2, n_words)
    epsilon = 1e-8

    assert y_true.shape[1] == 2
    assert y_pred.shape[1] == 2

    batch_size = y_true.shape[0]

    y_true_start, y_true_end = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

    p1 = tf.gather(params=y_pred_start, indices=y_true_start, axis=-1, batch_dims=-1)
    p2 = tf.gather(params=y_pred_end, indices=y_true_end, axis=-1, batch_dims=-1)

    p1 = tf.reshape(p1, shape=(batch_size, 1))
    p2 = tf.reshape(p2, shape=(batch_size, 1))

    log_p1 = tf.math.log(p1 + epsilon)
    log_p2 = tf.math.log(p2 + epsilon)

    neg_log_p1 = tf.math.negative(log_p1)
    neg_log_p2 = tf.math.negative(log_p2)

    sum = neg_log_p1 + neg_log_p2

    mean = tf.reduce_mean(sum)

    # tf.print('\nlog_p1: ', log_p1)
    # tf.print('\nlog_p2: ', log_p2)
    # tf.print("\nloss:", mean)

    return mean


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

    '''
    # Test if trains...
    w_context = np.random.randint(1, WORD_VOCAB_SIZE, (BATCH_SIZE, MAX_CONTEXT_WORDS))
    # Force some random padding in the input
    for row in range(w_context.shape[0]):
        n_pad = np.random.randint(0, 16)
        if n_pad > 0:
            w_context[row][-n_pad:] = 0

    c_context = np.random.randint(0, 100, (BATCH_SIZE, MAX_CONTEXT_WORDS, MAX_CHARS))

    w_query = np.random.randint(1, WORD_VOCAB_SIZE, (BATCH_SIZE, MAX_QUERY_WORDS))
    # Force some random padding in the input
    for row in range(w_query.shape[0]):
        n_pad = np.random.randint(0, 5)
        if n_pad > 0:
            w_query[row][-n_pad:] = 0

    c_query = np.random.randint(0, 100, (BATCH_SIZE, MAX_QUERY_WORDS, MAX_CHARS))

    labels = tf.random.uniform(shape=(BATCH_SIZE, 2, 1), minval=0, maxval=MAX_CONTEXT_WORDS, dtype=tf.int64)
    '''
    train_w_context, train_c_context, train_w_query, train_c_query, train_labels = generate_random_data(32)
    valid_w_context, valid_c_context, valid_w_query, valid_c_query, valid_labels = generate_random_data(8)

    history = model.fit(
        x={
            "context words": train_w_context,
            "context characters": train_c_context,
            "query words": train_w_query,
            "query characters": train_c_query
        },
        y=train_labels,
        validation_data=(
            {
                "context words": valid_w_context,
                "context characters": valid_c_context,
                "query words": valid_w_query,
                "query characters": valid_c_query
            }, valid_labels),
        verbose=1,
        batch_size=4,
        epochs=2)


if __name__ == '__main__':
    main()
