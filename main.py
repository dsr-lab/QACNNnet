import os

from model.question_answer_model import QACNNnet
import config
import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
from preprocessing.dataframe_builder import load_dataframe, build_embedding_matrix
from model.generator import Generator
import string

def load_data():
    '''
    Method responsible for loading the dataset, and returning it split as train
    and validation sets
    '''

    dataframe, words_tokenizer, chars_tokenizer, glove_dict = load_dataframe(force_rebuild=False)
    pretrained_embedding_weights = build_embedding_matrix(words_tokenizer, glove_dict)

    words_to_remove = ['a', 'an', 'the'] + list(string.punctuation)
    tokens_to_remove = []
    for w in words_to_remove:
        if w in words_tokenizer:
            tokens_to_remove.append(words_tokenizer[w])
    tokens_to_remove.append(0)  # Padding

    tokens_to_remove = tf.constant(tokens_to_remove)
    tokens_to_remove = tf.expand_dims(tokens_to_remove, -1)

    train_set = dataframe.loc[dataframe["Split"] == "train"]
    validation_set = dataframe.loc[dataframe["Split"] == "validation"]

    input_train = (np.stack(train_set["Context words"], axis=0),
                   np.stack(train_set["Context chars"], axis=0),
                   np.stack(train_set["Question words"], axis=0),
                   np.stack(train_set["Question chars"], axis=0))

    output_train = np.stack(train_set["Labels"], axis=0)

    input_validation = (np.stack(validation_set["Context words"], axis=0),
                        np.stack(validation_set["Context chars"], axis=0),
                        np.stack(validation_set["Question words"], axis=0),
                        np.stack(validation_set["Question chars"], axis=0))

    output_validation = np.stack(validation_set["Labels"], axis=0)

    config.config_model(len(words_tokenizer),
                        len(chars_tokenizer),
                        pretrained_embedding_weights,
                        tokens_to_remove)

    return input_train, input_validation, output_train, output_validation


# Build model and compile
def build_model(input_embedding_params, embedding_encoder_params, conv_input_projection_params,
                model_encoder_params, context_query_attention_params, output_params, max_context_words,
                max_query_words, max_chars, optimizer, vocab_size, ignore_tokens, dropout_rate):
    '''
    Build and compile the model by using all the configuration parameters and dictionaries
    received as argument.
    '''

    # Model input tensors
    context_words_input = tf.keras.Input(shape=(max_context_words), name="context words")
    context_characters_input = tf.keras.Input(shape=(max_context_words, max_chars), name="context characters")
    query_words_input = tf.keras.Input(shape=(max_query_words), name="query words")
    query_characters_input = tf.keras.Input(shape=(max_query_words, max_chars), name="query characters")

    inputs = [context_words_input, context_characters_input, query_words_input, query_characters_input]

    # Create the model and force a call
    model = QACNNnet(input_embedding_params,
                     embedding_encoder_params,
                     conv_input_projection_params,
                     model_encoder_params,
                     context_query_attention_params,
                     output_params,
                     vocab_size,
                     ignore_tokens,
                     dropout_rate)
    model(inputs)

    # Compile the model
    model.compile(
        # optimizer=LossScaleBelowOneOptimizer(optimizer),
        optimizer=mixed_precision.LossScaleOptimizer(optimizer),
        # optimizer,
        run_eagerly=config.EAGER_MODE
    )

    return model


def main():
    # tf.keras.utils.plot_model(model, "Architecture.png", show_shapes=True, expand_nested=True)
    policyConfig = 'mixed_float16'
    policy = tf.keras.mixed_precision.Policy(policyConfig)
    mixed_precision.set_global_policy(policy)

    input_train, input_validation, output_train, output_validation = load_data()

    if config.USE_GENERATOR:
        generator = Generator(input_train, output_train, input_validation, output_validation)

    valid_w_context, valid_c_context, valid_w_query, valid_c_query = input_validation
    if not config.USE_GENERATOR:
        train_w_context, train_c_context, train_w_query, train_c_query = input_train

    output_validation = np.expand_dims(output_validation, -1)
    output_train = np.expand_dims(output_train, -1)

    if config.DEBUG:
        config.BATCH_SIZE = 32
        n_val = 10
        valid_w_context = valid_w_context[:n_val]
        valid_c_context = valid_c_context[:n_val]
        valid_w_query = valid_w_query[:n_val]
        valid_c_query = valid_c_query[:n_val]
        output_validation = output_validation[:n_val]

        if not config.USE_GENERATOR:
            n_train = 50
            train_w_context = train_w_context[:n_train]
            train_c_context = train_c_context[:n_train]
            train_w_query = train_w_query[:n_train]
            train_c_query = train_c_query[:n_train]
            output_train = output_train[:n_train]

    if not config.USE_GENERATOR and config.TRAIN_ON_FULL_DATASET:
        train_w_context = np.concatenate((train_w_context, valid_w_context), axis=0)
        train_c_context = np.concatenate((train_c_context, valid_c_context), axis=0)
        train_w_query = np.concatenate((train_w_query, valid_w_query), axis=0)
        train_c_query = np.concatenate((train_c_query, valid_c_query), axis=0)
        output_train = np.concatenate((output_train, output_validation), axis=0)

    # Build the model
    model = build_model(config.input_embedding_params,
                        config.embedding_encoder_params,
                        config.conv_input_projection_params,
                        config.model_encoder_params,
                        config.context_query_attention_params,
                        config.output_params,
                        config.MAX_CONTEXT_WORDS,
                        config.MAX_QUERY_WORDS,
                        config.MAX_CHARS,
                        config.OPTIMIZER,
                        config.WORD_VOCAB_SIZE + 1,
                        config.IGNORE_TOKENS,
                        config.DROPOUT_RATE)

    print("Model succesfully built!")

    model.summary()

    # Load model weights if required
    if config.LOAD_WEIGHTS:
        if os.path.exists(config.CHECKPOINT_PATH + ".index"):
            print("Loading model's weights...")
            model.load_weights(config.CHECKPOINT_PATH)
            print("Model's weights successfully loaded!")

        else:
            print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
            print("Ignore this warning if it is a test.")

    # Add model checkpoint callbacks
    callbacks_list = []
    if config.SAVE_WEIGHTS:
        callbacks_list.append(
            tf.keras.callbacks.ModelCheckpoint(filepath=config.CHECKPOINT_PATH, save_weights_only=True, verbose=1))

    # Start the model training
    if config.USE_GENERATOR:
        history = model.fit(
            generator,
            validation_data=(
                [valid_w_context, valid_c_context,
                 valid_w_query, valid_c_query],
                output_validation) if not config.TRAIN_ON_FULL_DATASET else None,
            verbose=1,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            callbacks=callbacks_list)

    else:
        history = model.fit(
            x=[train_w_context, train_c_context, train_w_query, train_c_query],
            y=output_train,
            validation_data=(
                [valid_w_context, valid_c_context,
                 valid_w_query, valid_c_query],
                output_validation) if not config.TRAIN_ON_FULL_DATASET else None,
            verbose=1,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            callbacks=callbacks_list)

    return history, model


if __name__ == '__main__':
    history, model = main()
