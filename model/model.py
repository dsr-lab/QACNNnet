# This class is responsible for assembling the final model and defining training and testing info

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from question_answer_model import QACNNnet
from warmup_learning import CustomSchedule

#CONSTANTS

MAX_CONTEXT_WORDS = 400
MAX_QUERY_WORDS = 30

L2_VALUE =3e-7

WORD_EMBEDDING_SIZE = 300
WORD_VOCAB_SIZE = 10000 #TODO: take from preprocessing step
pretrained_weights = np.random.rand(WORD_VOCAB_SIZE, WORD_EMBEDDING_SIZE) #TODO: take from GloVe
CHARACTER_EMBEDDING_SIZE = 200
CHARACTER_VOCAB_SIZE = 100 #TODO: take from preprocessing step
MAX_CHARS = 16

EMBEDDING_KERNEL_SIZE = 5
N_HIGHWAY_LAYERS = 2

D_MODEL = 128
ENCODER_KERNEL_SIZE = 7
N_CONV_LAYERS_EMBEDDING_ENCODING = 4
N_CONV_LAYERS_MODEL_ENCODING = 2
N_HEADS = 8
STOCHASTIC_SURVIVAL_PROB = 0.9
N_BLOCKS_EMBEDDING_ENCODING = 1
N_BLOCKS_MODEL_ENCODING = 7

BATCH_SIZE = 32
EPOCHS = 100

#Learning rate, optimizer and loss

FINAL_LEARNING_RATE = 0.001
WARMUP_STEPS = 1000
learning_rate = CustomSchedule(FINAL_LEARNING_RATE, WARMUP_STEPS)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate, beta_1=0.8, beta_2=0.999, epsilon=1e-7)

#Layers' variables

input_embedding_params = {
"w_emb_size": WORD_EMBEDDING_SIZE,
"w_pretrained_weights": pretrained_weights,
"w_vocab_size": WORD_VOCAB_SIZE,
"w_n_special_tokens": 1,
"c_emb_size": CHARACTER_EMBEDDING_SIZE,
"c_vocab_size": CHARACTER_VOCAB_SIZE,
"c_conv_kernel_size": EMBEDDING_KERNEL_SIZE,
"n_highway_layers": N_HIGHWAY_LAYERS
}

embedding_encoder_params = {
"d_model": D_MODEL,
"kernel_size": ENCODER_KERNEL_SIZE,
"n_conv_layers": N_CONV_LAYERS_EMBEDDING_ENCODING,
"n_heads": N_HEADS,
"survival_prob": STOCHASTIC_SURVIVAL_PROB,
"l2_value": L2_VALUE,
"n_blocks": N_BLOCKS_EMBEDDING_ENCODING
}

conv_layer_params = {
"filters": D_MODEL,
"kernel_size": ENCODER_KERNEL_SIZE,
"padding": "same",
"data_format": "channels_last",
"kernel_regularizer": regularizers.l2(L2_VALUE)
}

model_encoder_params = {
"d_model": D_MODEL,
"kernel_size": ENCODER_KERNEL_SIZE,
"n_conv_layers": N_CONV_LAYERS_MODEL_ENCODING,
"n_heads": N_HEADS,
"survival_prob": STOCHASTIC_SURVIVAL_PROB,
"l2_value": L2_VALUE,
"n_blocks": N_BLOCKS_MODEL_ENCODING
}

def loss_function(y_true, y_pred):
    # y_true = (batch_size, 2, 1) or (batch_size, 2)
    # y_pred = (batch_size, 2, n_words)

    assert y_true.shape[1] == 2
    assert y_pred.shape[1] == 2

    batch_size = y_true.shape[0]

    y_true_start, y_true_end = tf.split(y_true, num_or_size_splits=2, axis=1)
    y_pred_start, y_pred_end = tf.split(y_pred, num_or_size_splits=2, axis=1)

    p1 = tf.gather(params=y_pred_start, indices=y_true_start, axis=-1, batch_dims=batch_size)
    p2 = tf.gather(params=y_pred_end, indices=y_true_end, axis=-1, batch_dims=batch_size)

    p1 = tf.reshape(p1, shape=(batch_size, 1))
    p2 = tf.reshape(p2, shape=(batch_size, 1))

    log_p1 = tf.math.log(p1)
    log_p2 = tf.math.log(p2)

    neg_log_p1 = tf.math.negative(log_p1)
    neg_log_p2 = tf.math.negative(log_p2)

    sum = neg_log_p1 + neg_log_p2

    mean = tf.reduce_mean(sum)

    return mean

#Build model and compile
def build_model (input_embedding_params, embedding_encoder_params, conv_layer_params, model_encoder_params, max_context_words, max_query_words, max_chars, optimizer, loss):

    net = QACNNnet(input_embedding_params, embedding_encoder_params, conv_layer_params, model_encoder_params)

    model = net.model(max_context_words, max_query_words, max_chars)

    model.compile(optimizer=optimizer, loss=loss)

    return model

model = build_model(input_embedding_params,
                    embedding_encoder_params,
                    conv_layer_params,
                    model_encoder_params,
                    MAX_CONTEXT_WORDS,
                    MAX_QUERY_WORDS,
                    MAX_CHARS,
                    OPTIMIZER,
                    loss_function)

print("Model succesfully built!")
print(model.summary())
#tf.keras.utils.plot_model(model, "Architecture.png", show_shapes=True, expand_nested=True)

#Test if trains...
w_context = np.random.randint(1, WORD_VOCAB_SIZE, (BATCH_SIZE, MAX_CONTEXT_WORDS))
# Force some random padding in the input
for row in range(w_context.shape[0]):
    n_pad = np.random.randint(0, 16)
    if n_pad > 0:
        w_context[row][-n_pad:] = 0
context_word_mask = w_context != 0
context_word_mask = tf.convert_to_tensor(context_word_mask)
c_context = np.random.randint(0, 100, (BATCH_SIZE, MAX_CONTEXT_WORDS, MAX_CHARS))

w_query = np.random.randint(1, WORD_VOCAB_SIZE, (BATCH_SIZE, MAX_QUERY_WORDS))
# Force some random padding in the input
for row in range(w_query.shape[0]):
    n_pad = np.random.randint(0, 5)
    if n_pad > 0:
        w_query[row][-n_pad:] = 0

query_word_mask = w_query != 0
query_word_mask = tf.convert_to_tensor(query_word_mask)
c_query = np.random.randint(0, 100, (BATCH_SIZE, MAX_QUERY_WORDS, MAX_CHARS))

labels = tf.random.uniform(shape=(BATCH_SIZE,2,2),minval=0, maxval=MAX_CONTEXT_WORDS, dtype=tf.int64)

history = model.fit(x={"context words": w_context, "context characters": c_context, "query words":w_query, "query characters":c_query},
                    y=labels,
                    verbose=1,
                    batch_size = 4)
