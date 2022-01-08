from metrics import qa_loss
from model import question_answer_model
from model.question_answer_model import QACNNnet
import Config
import tensorflow as tf
import numpy as np
from preprocessing.dataframe_builder import load_dataframe, build_embedding_matrix
import string


def load_data():

    dataframe, words_tokenizer, chars_tokenizer, glove_dict = load_dataframe(force_rebuild=False)
    pretrained_embedding_weights = build_embedding_matrix(words_tokenizer, glove_dict)

    Config.WORD_VOCAB_SIZE = len(words_tokenizer)
    Config.pretrained_weights = pretrained_embedding_weights
    Config.CHARACTER_VOCAB_SIZE = len(chars_tokenizer)

    words_to_remove = ['a', 'an', 'the'] + list(string.punctuation)
    tokens_to_remove = []
    for w in words_to_remove:
        if w in words_tokenizer:
            tokens_to_remove.append(words_tokenizer[w])

    tokens_to_remove = tf.constant(tokens_to_remove)
    tokens_to_remove = tf.expand_dims(tokens_to_remove, -1)
    Config.IGNORE_TOKENS = tokens_to_remove

    train_set = dataframe.loc[dataframe["Split"] == "train"]
    validation_set = dataframe.loc[dataframe["Split"] == "validation"]

    input_train = (np.stack(train_set["Context words"],axis=0),
    np.stack(train_set["Context chars"],axis=0),
    np.stack(train_set["Question words"],axis=0),
    np.stack(train_set["Question chars"],axis=0))

    output_train = np.stack(train_set["Labels"],axis=0)

    input_validation = (np.stack(validation_set["Context words"],axis=0),
    np.stack(validation_set["Context chars"],axis=0),
    np.stack(validation_set["Question words"],axis=0),
    np.stack(validation_set["Question chars"],axis=0))

    output_validation = np.stack(validation_set["Labels"],axis=0)

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
        run_eagerly=False,
    )

    return model


def generate_random_data(n_items):
    w_context = np.random.randint(1, Config.WORD_VOCAB_SIZE, (n_items, Config.MAX_CONTEXT_WORDS))
    # Force some random padding in the input
    for row in range(w_context.shape[0]):
        n_pad = np.random.randint(0, 16)
        if n_pad > 0:
            w_context[row][-n_pad:] = 0

    c_context = np.random.randint(0, 100, (n_items, Config.MAX_CONTEXT_WORDS, Config.MAX_CHARS))

    w_query = np.random.randint(1, Config.WORD_VOCAB_SIZE, (n_items, Config.MAX_QUERY_WORDS))
    # Force some random padding in the input
    for row in range(w_query.shape[0]):
        n_pad = np.random.randint(0, 5)
        if n_pad > 0:
            w_query[row][-n_pad:] = 0

    c_query = np.random.randint(0, 100, (n_items, Config.MAX_QUERY_WORDS, Config.MAX_CHARS))

    labels = tf.random.uniform(shape=(n_items, 2, 1), minval=0, maxval=Config.MAX_CONTEXT_WORDS, dtype=tf.dtypes.int64)

    return w_context, c_context, w_query, c_query, labels


def main():
    model = build_model(Config.input_embedding_params,
                        Config.embedding_encoder_params,
                        Config.conv_layer_params,
                        Config.model_encoder_params,
                        Config.MAX_CONTEXT_WORDS,
                        Config.MAX_QUERY_WORDS,
                        Config.MAX_CHARS,
                        Config.OPTIMIZER,
                        qa_loss)

    print("Model succesfully built!")
    model.summary()
    # tf.keras.utils.plot_model(model, "Architecture.png", show_shapes=True, expand_nested=True)

    input_train, input_validation, output_train, output_validation = load_data()
    train_w_context, train_c_context, train_w_query, train_c_query = input_train
    valid_w_context, valid_c_context, valid_w_query, valid_c_query = input_validation

    '''
    train_w_context = train_w_context[:500]
    train_c_context = train_c_context[:500]
    train_w_query = train_w_query[:500]
    train_c_query = train_c_query[:500]
    output_train = output_train[:500]

    valid_w_context = valid_w_context[:100]
    valid_c_context = valid_c_context[:100]
    valid_w_query = valid_w_query[:100]
    valid_c_query = valid_c_query[:100]
    output_validation = output_validation[:100]
    '''

    output_train = np.expand_dims(output_train, -1)
    output_validation = np.expand_dims(output_validation, -1)

    # train_w_context, \
    #     train_c_context, train_w_query, \
    #     train_c_query, output_train = generate_random_data(100)
    #
    # valid_w_context, \
    #     valid_c_context, valid_w_query, \
    #     valid_c_query, output_validation = generate_random_data(10)

    # TODO: refactoring required
    question_answer_model.f1_score.vocab_size = Config.WORD_VOCAB_SIZE + 1
    question_answer_model.em_score.vocab_size = Config.WORD_VOCAB_SIZE + 1
    question_answer_model.f1_score.ignore_tokens = Config.IGNORE_TOKENS
    question_answer_model.em_score.ignore_tokens = Config.IGNORE_TOKENS

    history = model.fit(
        x=[train_w_context, train_c_context, train_w_query, train_c_query],
        y=output_train,
        validation_data=(
            [valid_w_context, valid_c_context, valid_w_query, valid_c_query],
            output_validation),
        verbose=1,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS)

    return history


if __name__ == '__main__':
    history = main()
