import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import preprocess
import tokenizer
import glove_manager
import pickle
import Config

#TODO: Test A LOT (nothing tested yet, surely full of bugs and probably it does not even compile)

DATA_PATH = os.path.join("data", "training_set.json")
DATAFRAME_PATH = os.path.join("data", "training_dataframe.pkl")
WORDS_TOKENIZER_PATH = os.path.join("data", "words_tokenizer.pkl")
CHARS_TOKENIZER_PATH = os.path.join("data", "chars_tokenizer.pkl")

PREPROCESSING_OPTIONS = {
"strip":True,
"lower":True,
"replace":True,
"stopwords":False,
"lemmatize":True
}

TRAIN_SAMPLES = 90000

np.random.seed(seed=100) #Define a seed for randomization, avoiding to get different placeholder or random embeddings each time
UNK_PLACEHOLDER = np.random.uniform(low=-0.05, high=0.05, size=glove_manager.EMBEDDING_SIZE)

def get_data(path):

    absolute_data_path = os.path.join(os.getcwd(), path)

    with open(absolute_data_path, "r") as file:
        data = json.loads(file.read())

    return data

def get_answer_indices(context_words, answer_words):

    i = 0
    for j, context_word in enumerate(context_words):
        if context_word == answer_words[i]:
            i+=1
            if i==len(answer_words):
                start = j - len(answer_words) + 1
                end = j
                return [start,end]
        else:
            i=0

def build_dataframe_row(context, question, answer, split):

    preprocessed_context = preprocess.preprocess_text(context, **PREPROCESSING_OPTIONS)
    preprocessed_question = preprocess.preprocess_text(question, **PREPROCESSING_OPTIONS)
    preprocessed_answer = preprocess.preprocess_text(answer, **PREPROCESSING_OPTIONS)

    answer_indices = get_answer_indices(preprocessed_context, preprocessed_answer)

    preprocessed_context_chars = [preprocess.split_to_chars(word) for word in preprocessed_context]
    preprocessed_question_chars = [preprocess.split_to_chars(word) for word in preprocessed_question]

    row = {
    "Context words":preprocessed_context,
    "Context chars":preprocessed_context_chars,
    "Question words":preprocessed_question,
    "Question chars":preprocessed_question_chars,
    "Labels":answer_indices,
    "Split":split
    }

    return row

def extract_rows(json_dict):

    print("Data extraction started...")

    version = json_dict["version"] #Not used

    data = json_dict["data"]

    dataframe_rows = []

    for element in tqdm(data):
        title = element["title"]
        paragraphs = element["paragraphs"]

        for paragraph in paragraphs:
            context = paragraph["context"]
            questions_answers = paragraph["qas"]

            for qas in questions_answers:
                question = qas["question"]
                id = qas["id"] #Not used
                answers = qas["answers"]

                for answer in answers:
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"] #Not used

                    split = "train" if len(dataframe_rows) < TRAIN_SAMPLES else "validation"

                    dataframe_rows.append(build_dataframe_row(context, question, answers_text, split))

    print("Data extraction completed!"")

    return dataframe_rows

def tokenize_dataframe(df, words_tokenizer, chars_tokenizer):

    df["Context words"] = df["Context words"].apply(lambda words: tokenizer.pad_truncate_tokenize_words(words, words_tokenizer, Config.MAX_CONTEXT_WORDS))
    df["Question words"] = df["Question words"].apply(lambda words: tokenizer.pad_truncate_tokenize_words(words, words_tokenizer, Config.MAX_QUERY_WORDS))
    df["Context chars"] = df["Context chars"].apply(lambda chars: tokenizer.pad_truncate_tokenize_chars_sequence(chars,chars_tokenizer,Config.MAX_CONTEXT_WORDS,Config.MAX_CHARS))
    df["Question chars"] = df["Question chars"].apply(lambda chars: tokenizer.pad_truncate_tokenize_chars_sequence(chars,chars_tokenizer,Config.MAX_QUERY_WORDS,Config.MAX_CHARS))

    return df

def build_dataframe():

    data = get_data(DATA_PATH)
    dataframe_rows = extract_rows(data)

    print("Tokenization started...")

    glove_manager.setup_files()
    glove_dict = glove_manager.load_glove()

    unique_words = tokenizer.get_unique_words(dataframe_rows)
    unique_chars = tokenizer.get_unique_chars(dataframe_rows)

    words_tokenizer = tokenizer.build_words_tokenizer(unique_words, glove_dict)
    chars_tokenizer = tokenizer.build_chars_tokenizer(unique_chars)

    dataframe = pd.DataFrame(dataframe_rows)
	dataframe = dataframe[["Context words", "Context chars", "Question words", "Question chars", "Labels", "Split"]]

    dataframe = tokenize_dataframe(dataframe, words_tokenizer, chars_tokenizer)

    print("Tokenization completed!")

    return dataframe, words_tokenizer, chars_tokenizer, glove_dict

def save_dataframe(dataframe, words_tokenizer, chars_tokenizer):

    dataframe.to_pickle(DATAFRAME_PATH)
    with open(WORDS_TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(words_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(CHARS_TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(chars_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Dataframe saved successfully")

def check_savings(paths):
    for path in paths:
        if not os.path.exists(path):
            return False

    return True

def load_dataframe(force_rebuild=False):

    if not check_savings(DATAFRAME_PATH, WORDS_TOKENIZER_PATH, CHARS_TOKENIZER_PATH) or force_rebuild:
		dataframe, words_tokenizer, chars_tokenizer, glove_dict = build_dataframe()
        save_dataframe(dataframe)
        return dataframe, words_tokenizer, chars_tokenizer, glove_dict

    else:
        dataframe = pd.read_pickle(DATAFRAME_PATH)
        with open(WORDS_TOKENIZER_PATH, 'rb') as handle:
            words_tokenizer = pickle.load(handle)
        with open(CHARS_TOKENIZER_PATH, 'rb') as handle:
            chars_tokenizer = pickle.load(handle)
        glove_manager.setup_files()
        glove_dict = glove_manager.load_glove()
        return dataframe, words_tokenizer, chars_tokenizer, glove_dict

def build_embedding_matrix(words_tokenizer, glove_dict):

    vocab_size = len(words_tokenizer)

    embedding_matrix = np.zeros((vocab_size, glove_manager.EMBEDDING_SIZE), dtype=np.float32)

    print("Building embedding matrix started...")

    for word, token in tqdm(words_tokenizers.items()):
        if word in glove_dict:
            embedding_matrix[token-1] = glove_dict[word]
        else:
            embedding_matrix[token-1] = UNK_PLACEHOLDER

    print("Building embedding matrix completed!")

    return embedding_matrix
