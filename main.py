from metrics import qa_loss
from model.question_answer_model import QACNNnet
from Config import *
from preprocessing.dataframe_builder import load_dataframe, build_embedding_matrix

def load_data():

    dataframe, words_tokenizer, chars_tokenizer, glove_dict = load_dataframe()
    pretrained_embedding_weights = build_embedding_matrix(words_tokenizer, glove_dict)

    #TODO: set pretrained weights from config and also vocab's sizes

    train_set = dataframe.loc[dataframe["Split"] == "train"]
    validation_set = dataframe.loc[dataframe["Split"] == "validation"]

    input_train = (np.array(train_set["Context words"]),
    np.array(train_set["Context chars"]),
    np.array(train_set["Question words"]),
    np.array(train_set["Question chars"]))

    output_train = np.array(train_set["Labels"])

    input_validation = (np.array(validation_set["Context words"]),
    np.array(validation_set["Context chars"]),
    np.array(validation_set["Question words"]),
    np.array(validation_set["Question chars"]))

    output_validation = np.array(validation_set["Labels"])

    return input_train, input_validation, output_train, output_validation

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
        run_eagerly=True,
        metrics=[]
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
                        qa_loss)

    print("Model succesfully built!")
    model.summary()
    # tf.keras.utils.plot_model(model, "Architecture.png", show_shapes=True, expand_nested=True)

    train_w_context, \
    train_c_context, train_w_query, \
    train_c_query, train_labels = generate_random_data(32)

    valid_w_context, \
    valid_c_context, valid_w_query, \
    valid_c_query, valid_labels = generate_random_data(8)

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


if __name__ == '__main__':

    main()
