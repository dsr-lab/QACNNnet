import os
import sys
import json
from tqdm import tqdm
import pickle
import pandas as pd
import preprocessing.glove_manager as glove_manager
import preprocessing.preprocess as preprocess
from preprocessing.dataframe_builder import tokenize_dataframe, build_embedding_matrix
from model.question_answer_model import QACNNnet
from inference import get_predictions
import numpy as np
import tensorflow as tf
import string
import config


class PredLayer(tf.keras.layers.Layer):
    '''
    A custom layer conveniently used just for splitting the start and end predictions.
    This is not part of the actual model, and it is specifically used in this script only for
    creating the prediction.json file required by the evaluation script.
    '''

    def call(self, input):
        raw_predictions_start, raw_predictions_end = tf.split(input, num_or_size_splits=2, axis=1)
        pred_start, pred_end = get_predictions(raw_predictions_start, raw_predictions_end)

        return pred_start, pred_end


def load_dictionaries():
    '''
    Load tokenizers and GLoVe
    '''

    with open(config.WORDS_TOKENIZER_PATH, 'rb') as handle:
        words_tokenizer = pickle.load(handle)
    with open(config.CHARS_TOKENIZER_PATH, 'rb') as handle:
        chars_tokenizer = pickle.load(handle)

    glove_manager.setup_files()
    glove_dict = glove_manager.load_glove()

    return words_tokenizer, chars_tokenizer, glove_dict


def build_dataframe_row(context, question, title, id, corpus):
    '''
    Methods that take as input a context, a question, a title, an identifier and a corpus
    that is filled every time a new row is processed.
    '''

    # Preprocess the context words and append it to the corpus
    preprocessed_context, full_text = preprocess.preprocess_text(context, config.PREPROCESSING_OPTIONS, get_full_text=True)
    corpus.append(full_text)

    # Limit the max length
    if len(preprocessed_context)>config.MAX_CONTEXT_WORDS:
        preprocessed_context = preprocessed_context[0:config.MAX_CONTEXT_WORDS]

    # Preprocess the question words
    preprocessed_question = preprocess.preprocess_text(question, config.PREPROCESSING_OPTIONS)

    # Preprocess the context and question chars
    preprocessed_context_chars = [preprocess.split_to_chars(word) for word in preprocessed_context]
    preprocessed_question_chars = [preprocess.split_to_chars(word) for word in preprocessed_question]

    # Create the dataframe row
    row = {
        "Title": title,
        "Question ID":id,
        "Context words":preprocessed_context,
        "Context chars":preprocessed_context_chars,
        "Question words":preprocessed_question,
        "Question chars":preprocessed_question_chars,
    }

    return row


def extract_rows(json_dict):
    '''
    Method responsible for creating the dataframe rows from the original json dataset
    '''

    print("Data extraction started...")

    data = json_dict["data"]

    dataframe_rows = []
    corpus = []

    for element in tqdm(data):
        title = element["title"]
        paragraphs = element["paragraphs"]

        for paragraph in paragraphs:
            context = paragraph["context"]
            questions_answers = paragraph["qas"]

            for qas in questions_answers:
                question = qas["question"]
                id = qas["id"]

                row = build_dataframe_row(context, question, title, id, corpus)
                dataframe_rows.append(row)

    print("Data extraction completed!")

    return dataframe_rows, corpus


def build_dataframe(data_path):
    '''
    Method that creates the DataFrame containing all the information stored in the json
    that is at the path passed as argument.
    '''

    with open(data_path, "r") as file:
        data = json.loads(file.read())
    dataframe_rows, corpus = extract_rows(data)

    print("Tokenization started...")

    words_tokenizer, chars_tokenizer, glove_dict = load_dictionaries()

    dataframe = pd.DataFrame(dataframe_rows)
    dataframe = dataframe[["Title", "Question ID" ,"Context words", "Context chars", "Question words", "Question chars"]]

    dataframe = tokenize_dataframe(dataframe, words_tokenizer, chars_tokenizer)

    print("Tokenization completed!")

    return dataframe, words_tokenizer, chars_tokenizer, glove_dict, corpus


def load_data(data_path):
    '''
    Wrapper method that calls all the sub-methods for creating the dataset usable
    by the model
    '''

    dataframe, words_tokenizer, chars_tokenizer, glove_dict, corpus = build_dataframe(data_path)
    pretrained_embedding_weights = build_embedding_matrix(words_tokenizer, glove_dict)

    words_to_remove = ['a', 'an', 'the'] + list(string.punctuation)
    tokens_to_remove = []
    for w in words_to_remove:
        if w in words_tokenizer:
            tokens_to_remove.append(words_tokenizer[w])
    tokens_to_remove.append(0)  # Padding

    tokens_to_remove = tf.constant(tokens_to_remove)
    tokens_to_remove = tf.expand_dims(tokens_to_remove, -1)

    input_test = (np.stack(dataframe["Context words"],axis=0),
    np.stack(dataframe["Context chars"],axis=0),
    np.stack(dataframe["Question words"],axis=0),
    np.stack(dataframe["Question chars"],axis=0))

    question_ids = np.stack(dataframe["Question ID"],axis=0)

    config.config_model(len(words_tokenizer),
                        len(chars_tokenizer),
                        pretrained_embedding_weights,
                        tokens_to_remove)

    return input_test, question_ids, words_tokenizer, corpus


def build_model(input_embedding_params, embedding_encoder_params, conv_input_projection_params,
                model_encoder_params, context_query_attention_params, output_params, max_context_words,
                max_query_words, max_chars, optimizer, vocab_size, ignore_tokens, dropout_rate):
    '''
    Method that has been also defined in the main.py, and replicated here for convenience.
    This is the method responsible for creating and compiling the tensorflow model.
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
        optimizer=optimizer
    )

    return model


def get_preprocessed_answers(contexts, pred_start, pred_end, tokenizer):
    '''
    Method that creates an array of preprocessed answers.
    The preprocessing applied is the same that has been used during the training of the model.
    '''

    pred_slices = tf.concat([pred_start, pred_end],-1).numpy()

    preprocessed_answers = []
    for context, pred_slice in zip(contexts, pred_slices):
        sliced_context = context[pred_slice[0]:pred_slice[1]+1]
        preprocessed_answers.append(" ".join(sliced_context))

    return preprocessed_answers


def write_answers(question_ids, answers):
    '''
    Write the answers in the predictions.json file, that will be ultimately used by
    the evaluation script.
    '''

    assert len(question_ids)==len(answers)

    answers_dict = {id:answer for id,answer in zip(question_ids,answers)}

    with open(config.PREDICTIONS_PATH, 'w') as json_file:
        json.dump(answers_dict, json_file)


def run_predictions(data_path):
    '''
    Create the model by restoring the training weights, and then perform a model.predict()
    on the dataset whose path is passed as argument of this function.
    '''

    input_test, question_ids, words_tokenizer, corpus = load_data(data_path)
    test_w_context, test_c_context, test_w_query, test_c_query = input_test

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

    if os.path.exists(config.CHECKPOINT_PATH + ".index"):
        print("Loading model's weights...")
        model.load_weights(config.CHECKPOINT_PATH)
        print("Model's weights successfully loaded!")

    else:
        print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
        print("Ignore this warning if it is a test.")

    print("Model succesfully built!")

    print("Starting predicting...")

    # Get the output predictions from the originally trained model
    raw_predictions = model.predict(x=[test_w_context, test_c_context, test_w_query, test_c_query],
                                    batch_size=config.BATCH_SIZE,
                                    verbose=1)

    # Create a new fake model, composed just by one layer, whose goal is to conveniently
    # convert the raw_predictions to predictions that leverage the inferencea algorithm
    # (see the relation for more details)
    inputs = tf.keras.Input(shape=(2, config.MAX_CONTEXT_WORDS))
    outputs = PredLayer()(inputs)

    refine_pred_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    pred_start, pred_end = refine_pred_model.predict(raw_predictions, batch_size=config.BATCH_SIZE, verbose=1)

    # Get the answers from out of the model predictions
    preprocessed_answers = get_preprocessed_answers(corpus, pred_start, pred_end, words_tokenizer)

    print("Predictions completed!")

    # Write the results to the prediction.json file
    write_answers(question_ids.tolist(), preprocessed_answers)

    print("Predictions successfully written to file.")


# Main:
args = sys.argv
if len(args)==2:
    data_path = args[1]
    if os.path.exists(data_path):
        run_predictions(data_path)
    else:
        print("Invalid argument: {} does not exists".format(data_path))
elif len(args)<2:
    print("Missing one required argument: 'test set path'")
else:
    print("Too many arguments, only one is expected: 'test set path'")
