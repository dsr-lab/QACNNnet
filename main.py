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
    )

    return model


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
