import numpy as np
import pandas as pd
from metrics_extractor import extract_metrics
import preprocessing.preprocess as preprocess
import preprocessing.tokenizer as tokenizer
import preprocessing.glove_manager as glove_manager
import config

#This module is used to create the dataset used by the question classifier

F1_ERROR_THRESHOLD = 0.5 #Define the minimum F1 score to consider a question partially solved
CLASSIFICATION_LABELS = {"Easy":[1,0,0], "Medium":[0,1,0],"Difficult":[0,0,1]} #One-hot encoding of classes

np.random.seed(seed=100) #Define a seed for randomization, avoiding to get different placeholder or random embeddings each time
UNK_PLACEHOLDER = np.random.uniform(low=-0.05, high=0.05, size=glove_manager.EMBEDDING_SIZE) #Random initial embedding used for the UNK token

def classify_into_label(question,exact,f1):

    '''
    classifiy questions into one of the following classes:
    -Easy (EM=1)
    -Medium (F1>threshold)
    -Difficult (F1<threshold and EM=0)
    '''

    if exact==1.0:
        return CLASSIFICATION_LABELS["Easy"]
    elif f1>F1_ERROR_THRESHOLD:
        return CLASSIFICATION_LABELS["Medium"]
    else:
        return CLASSIFICATION_LABELS["Difficult"]

def extract_row(question, val):

    '''
    Build a row (for the dataframe) containing the preprocessed question texts
    (split into tokens) and the correspondent labels.
    '''

    exact = val["EM"]
    f1 = val["F1"]

    label = classify_into_label(question, exact, f1)
    #Use the same preprocessing adopted fot the main model for coherence
    preprocessed_question = preprocess.preprocess_text(question, config.PREPROCESSING_OPTIONS)

    row = {
    "Question":preprocessed_question,
    "Label":label
    }

    return row

def get_unique_words(text_rows):

    '''
    Get a set of unique words in a list of texts.
    '''

    unique_words = set()
    for row in text_rows:
        question_words = set(row["Question"])

        unique_words = unique_words | question_words

    return unique_words

def build_embedding_matrix(words_tokenizer, glove_dict):

    '''
    Return the embedding matrix based on GloVe embeddings.
    '''

    vocab_size = len(words_tokenizer)+1

    # Initialize matrix, padding is considered
    embedding_matrix = np.zeros((vocab_size, glove_manager.EMBEDDING_SIZE), dtype=np.float16)

    print("Building embedding matrix started...")

    # Fill matrix with Glove's embeddings
    for word, token in words_tokenizer.items():
        if word in glove_dict:
            embedding_matrix[token] = glove_dict[word]
        else:
            embedding_matrix[token] = UNK_PLACEHOLDER

    print("Building embedding matrix completed!")

    return embedding_matrix

def build_dataframe(true_ans_path, pred_ans_path):

    '''
    Build the dataframe used for question classification.
    '''

    #Get the scores from official the evaluation script
    scores = extract_metrics(true_ans_path, pred_ans_path)

    print("Building dataframe...")

    #Build dataframe rows
    df_rows = []
    for question, val in scores.items():
        df_rows.append(extract_row(question,val))

    print("Dataframe succesfully built")

    #Load GloVe
    glove_manager.setup_files()
    glove_dict = glove_manager.load_glove()

    print("Tokenization started...")

    #Tokenize the dataframe questions
    unique_words = get_unique_words(df_rows)
    words_tokenizer = tokenizer.build_words_tokenizer(unique_words, glove_dict)

    df = pd.DataFrame(df_rows)
    df = df[["Question","Label"]]

    df["Question"] = df["Question"].apply(lambda words: tokenizer.pad_truncate_tokenize_words(words, words_tokenizer, config.MAX_QUERY_WORDS))

    print("Tokenization completed!")

    embedding_matrix = build_embedding_matrix(words_tokenizer, glove_dict)

    return df, embedding_matrix, words_tokenizer
