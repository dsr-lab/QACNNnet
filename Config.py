import Config
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from model.warmup_learning import CustomSchedule

DEBUG = False
EAGER_MODE = False

MAX_CONTEXT_WORDS = 400
MAX_QUERY_WORDS = 30
MAX_ANSWER_LENGTH = 30

L2_VALUE = 3e-7

IGNORE_TOKENS = tf.constant([[0], [1], [9], [10]])

WORD_EMBEDDING_SIZE = 300
WORD_VOCAB_SIZE = 10000  # TODO: take from preprocessing step
pretrained_weights = np.random.rand(WORD_VOCAB_SIZE, WORD_EMBEDDING_SIZE)  # TODO: take from GloVe
CHARACTER_EMBEDDING_SIZE = 96
CHARACTER_VOCAB_SIZE = 100  # TODO: take from preprocessing step
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
N_BLOCKS_MODEL_ENCODING = 7

BATCH_SIZE = 32
EPOCHS = 20

# Learning rate, optimizer and loss

FINAL_LEARNING_RATE = 0.001
WARMUP_STEPS = 1000.0
learning_rate = CustomSchedule(FINAL_LEARNING_RATE, WARMUP_STEPS)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate, beta_1=0.8, beta_2=0.999, epsilon=1e-7)

# Layers' variables

input_embedding_params = {}

embedding_encoder_params = {}

conv_layer_params = {}

model_encoder_params = {}


def config_model(word_vocab_size, char_vocab_size, pretrained_weights, ignore_tokens):
    Config.WORD_VOCAB_SIZE = word_vocab_size
    Config.CHARACTER_VOCAB_SIZE = char_vocab_size
    Config.pretrained_weights = pretrained_weights
    Config.IGNORE_TOKENS = ignore_tokens

    Config.input_embedding_params = {
        "w_emb_size": WORD_EMBEDDING_SIZE,
        "w_pretrained_weights": pretrained_weights,
        "w_vocab_size": WORD_VOCAB_SIZE,
        "w_n_special_tokens": 1,
        "c_emb_size": CHARACTER_EMBEDDING_SIZE,
        "c_vocab_size": CHARACTER_VOCAB_SIZE,
        "c_conv_kernel_size": EMBEDDING_KERNEL_SIZE,
        "n_highway_layers": N_HIGHWAY_LAYERS
    }

    Config.embedding_encoder_params = {
        "d_model": D_MODEL,
        "kernel_size": ENCODER_KERNEL_SIZE,
        "n_conv_layers": N_CONV_LAYERS_EMBEDDING_ENCODING,
        "n_heads": N_HEADS,
        "survival_prob": STOCHASTIC_SURVIVAL_PROB,
        "l2_value": L2_VALUE,
        "n_blocks": N_BLOCKS_EMBEDDING_ENCODING
    }

    Config.conv_layer_params = {
        "filters": D_MODEL,
        "kernel_size": ENCODER_KERNEL_SIZE,
        "padding": "same",
        "data_format": "channels_last",
        "kernel_regularizer": regularizers.l2(L2_VALUE)
    }

    Config.model_encoder_params = {
        "d_model": D_MODEL,
        "kernel_size": ENCODER_KERNEL_SIZE,
        "n_conv_layers": N_CONV_LAYERS_MODEL_ENCODING,
        "n_heads": N_HEADS,
        "survival_prob": STOCHASTIC_SURVIVAL_PROB,
        "l2_value": L2_VALUE,
        "n_blocks": N_BLOCKS_MODEL_ENCODING
    }