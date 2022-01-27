import sys
import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from preprocessing.question_dataframe_builder import build_dataframe
import config

TRAIN_PROPORTION = 0.9
VAL_PROPORTION = 0.05
TEST_PROPORTION = 0.05

RNN_SIZE = 64

BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.1

def extract_parameters(args):

    if len(args)==3:
        true_ans_path = args[1]
        pred_ans_path = args[2]
        if not os.path.exists(true_ans_path):
            print("Invalid argument: {} does not exists".format(true_ans_path))
            return None,None
        elif not os.path.exists(pred_ans_path):
            print("Invalid argument: {} does not exists".format(pred_ans_path))
            return None,None
        else:
            return true_ans_path, pred_ans_path

    elif len(args)<3:
        print("Missing one or more required argument: 'test set path' and 'predictions path' required")
        return None,None

    else:
        print("Too many arguments, two are expected: 'test set path' and 'predictions path'")
        return None,None

def print_distribution_info(y, name):

    easy = y[y==[1,0,0]]
    medium = y[y==[0,1,0]]
    difficult = y[y==[0,0,1]]

    print("Labels distribution in "+name+":")
    print("Easy instances: {}".format(easy.shape[0]))
    print("Medium instances: {}".format(medium.shape[0]))
    print("Difficult instances: {}".format(difficult.shape[0]))
    print()

def get_sets(df, show_distribution=True):

    other_sets = df.sample(frac=VAL_PROPORTION+TEST_PROPORTION, axis=0)
    training_set = df.drop(index=other_sets.index)
    validation_set = df.sample(frac=0.5, axis=0)
    test_set = df.drop(index=validation_set.index)

    X_train = np.stack(training_set["Question"],axis=0)
    y_train = np.stack(training_set["Label"],axis=0)

    X_val = np.stack(validation_set["Question"],axis=0)
    y_val = np.stack(validation_set["Label"],axis=0)

    X_test = np.stack(test_set["Question"],axis=0)
    y_test = np.stack(test_set["Label"],axis=0)

    if show_distribution:
        print_distribution_info(y_train,"training set")
        print_distribution_info(y_val,"validation set")
        print_distribution_info(y_test,"test set")

    return X_train,y_train,X_val,y_val,X_test,y_test

def build_model(dataframe, embedding_matrix, vocab_size):

    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size+1,
            config.WORD_EMBEDDING_SIZE,
            input_length=config.MAX_QUERY_WORDS,
            mask_zero=True,
            trainable=False,
            weights=[embedding_matrix]))
    model.add(layers.Bidirectional(layers.LSTM(RNN_SIZE, dropout=DROPOUT_RATE)))
    #model.add(layers.Dense(256, activation="tanh"))
    model.add(layers.Dense(3, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()

    return model

def train_model(model, X_train, y_train, X_val, y_val):

    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        verbose=1,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS)

    return history, model

def show_history(history):
    pass

def test_model(model, X_test, y_test):

    results = model.evaluate(x=X_test,
                    y=y_test,
                    batch_size=BATCH_SIZE,
                    verbose=1)

    return results

def run_model(true_ans_path, pred_ans_path):

    dataframe, embedding_matrix, words_tokenizer = build_dataframe(true_ans_path, pred_ans_path)
    vocab_size=len(words_tokenizer)

    X_train,y_train,X_val,y_val,X_test,y_test = get_sets(dataframe)

    model = build_model(dataframe, embedding_matrix, vocab_size)

    history, model = train_model(model, X_train, y_train, X_val, y_val)

    show_history(history)

    results = test_model(model, X_test, y_test)

    print()
    print("Loss on test set: {}".format(results[0]))
    print("Accuracy on test set: {}".format(results[1]))

#Main
args = sys.argv
true_ans_path, pred_ans_path = extract_parameters(args)
if true_ans_path is not None and pred_ans_path is not None:
    run_model(true_ans_path, pred_ans_path)
