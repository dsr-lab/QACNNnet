import os
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
from model.warmup_learning import CustomSchedule
import Config

DEBUG = True
EAGER_MODE = False
SAVE_WEIGHTS = True
LOAD_WEIGHTS = True

DATA_PATH = os.path.join("data", "training_set.json")
DATAFRAME_PATH = os.path.join("data", "training_dataframe.pkl")
WORDS_TOKENIZER_PATH = os.path.join("data", "words_tokenizer.pkl")
CHARS_TOKENIZER_PATH = os.path.join("data", "chars_tokenizer.pkl")
CHECKPOINT_PATH = os.path.join("data","Checkpoints","weights.ckpt")
PREDICTIONS_PATH = os.path.join("predictions", "predictions.json")

PREPROCESSING_OPTIONS = {
"strip":True,
"lower":True,
"replace":True,
"remove special":False,
"stopwords":False,
"lemmatize":False
}

TRAIN_SAMPLES = 78000

MAX_CONTEXT_WORDS = 400
MAX_QUERY_WORDS = 30
MAX_ANSWER_LENGTH = 30

L2_RATE = 3e-7
DROPOUT_RATE = 0.1

IGNORE_TOKENS = tf.constant([[0], [1], [9], [10]])

WORD_EMBEDDING_SIZE = 100
WORD_VOCAB_SIZE = 10000
PRETRAINED_WEIGHTS = np.random.rand(WORD_VOCAB_SIZE, WORD_EMBEDDING_SIZE)
CHARACTER_EMBEDDING_SIZE = 94
CHARACTER_VOCAB_SIZE = 100
MAX_CHARS = 16

EMBEDDING_KERNEL_SIZE = 5
N_HIGHWAY_LAYERS = 2

D_MODEL = 128
ENCODER_KERNEL_SIZE = 7
N_CONV_LAYERS_EMBEDDING_ENCODING = 4
N_CONV_LAYERS_MODEL_ENCODING = 2
N_HEADS = 1
STOCHASTIC_SURVIVAL_PROB = 0.9
N_BLOCKS_EMBEDDING_ENCODING = 1
N_BLOCKS_MODEL_ENCODING = 3

BATCH_SIZE = 32
EPOCHS = 30

# Learning rate, optimizer and loss
FINAL_LEARNING_RATE = 0.001
WARMUP_STEPS = 1000.0
learning_rate = CustomSchedule(FINAL_LEARNING_RATE, WARMUP_STEPS)

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate, beta_1=0.8, beta_2=0.999, epsilon=1e-7)
# OPTIMIZER = tfa.optimizers.MovingAverage(OPTIMIZER)

# Layers' variables

input_embedding_params = {}

embedding_encoder_params = {
    "d_model": D_MODEL,
    "kernel_size": ENCODER_KERNEL_SIZE,
    "n_conv_layers": N_CONV_LAYERS_EMBEDDING_ENCODING,
    "n_heads": N_HEADS,
    "survival_prob": STOCHASTIC_SURVIVAL_PROB,
    "n_blocks": N_BLOCKS_EMBEDDING_ENCODING,
    "dropout_rate": DROPOUT_RATE,
    "l2_rate": L2_RATE
}

conv_input_projection_params = {
    "filters": D_MODEL,
    "kernel_size": 1,
    "kernel_regularizer": None if L2_RATE == 0.0 else tf.keras.regularizers.l2(L2_RATE),
    # "activity_regularizer": tf.keras.regularizers.l2(L2_RATE),
    "bias_regularizer": None if L2_RATE == 0.0 else tf.keras.regularizers.l2(L2_RATE)

}

model_encoder_params = {
    "d_model": D_MODEL,
    "kernel_size": ENCODER_KERNEL_SIZE,
    "n_conv_layers": N_CONV_LAYERS_MODEL_ENCODING,
    "n_heads": N_HEADS,
    "survival_prob": STOCHASTIC_SURVIVAL_PROB,
    "n_blocks": N_BLOCKS_MODEL_ENCODING,
    "dropout_rate": DROPOUT_RATE,
    "l2_rate": L2_RATE
}

context_query_attention_params = {
    "dropout_rate": DROPOUT_RATE,
    "l2_rate": L2_RATE
}


def config_model(word_vocab_size, char_vocab_size, pretrained_weights, ignore_tokens):
    Config.WORD_VOCAB_SIZE = word_vocab_size
    Config.CHARACTER_VOCAB_SIZE = char_vocab_size
    Config.PRETRAINED_WEIGHTS = pretrained_weights
    Config.IGNORE_TOKENS = ignore_tokens

    Config.input_embedding_params = {
        "w_emb_size": WORD_EMBEDDING_SIZE,
        "w_pretrained_weights": PRETRAINED_WEIGHTS,
        "w_vocab_size": WORD_VOCAB_SIZE,
        "w_n_special_tokens": 1,
        "c_emb_size": CHARACTER_EMBEDDING_SIZE,
        "c_vocab_size": CHARACTER_VOCAB_SIZE,
        "c_conv_kernel_size": EMBEDDING_KERNEL_SIZE,
        "c_conv_output_size": D_MODEL,
        "n_highway_layers": N_HIGHWAY_LAYERS,
        "dropout_rate": DROPOUT_RATE,
        "l2_rate": L2_RATE,
        "conv_input_projection_params": conv_input_projection_params
    }
