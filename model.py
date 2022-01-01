# This class is responsible for assembling the final model and defining training and testing info

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from context_query_attention import ContextQueryAttentionLayer
from encoding.encoder import Encoding_Layer, EncoderLayer
from input_embedding.input_embedding_layer import InputEmbeddingLayer
from model_output import OutputLayer

BATCH_SIZE = 32
N_CONTEXT = 400
N_QUERY = 30
W_EMB_SIZE = 300
C_EMB_SIZE = 200
C_VOCAB_SIZE = 100
W_VOCAB_SIZE = 10000
W_SPECIAL_TOKEN = 1
MAX_CHAR = 16


def test_model():
    training = False

    w_context = np.random.randint(1, W_VOCAB_SIZE, (BATCH_SIZE, N_CONTEXT))

    claims_input = tf.keras.Input(shape=(N_CONTEXT), name="context")
    evidences_input = tf.keras.Input(shape=(N_QUERY), name="query")

    # Force some random padding in the input
    for row in range(w_context.shape[0]):
        n_pad = np.random.randint(0, 16)
        if n_pad > 0:
            w_context[row][-n_pad:] = 0
    context_word_mask = w_context != 0
    context_word_mask = tf.convert_to_tensor(context_word_mask)
    c_context = np.random.randint(0, 100, (BATCH_SIZE, N_CONTEXT, MAX_CHAR))

    w_query = np.random.randint(1, W_VOCAB_SIZE, (BATCH_SIZE, N_QUERY))
    # Force some random padding in the input
    for row in range(w_query.shape[0]):
        n_pad = np.random.randint(0, 5)
        if n_pad > 0:
            w_query[row][-n_pad:] = 0

    query_word_mask = w_query != 0
    query_word_mask = tf.convert_to_tensor(query_word_mask)
    c_query = np.random.randint(0, 100, (BATCH_SIZE, N_QUERY, MAX_CHAR))

    w_emb_weights = np.random.rand(W_VOCAB_SIZE, W_EMB_SIZE)

    # 1. Embedding
    inputEmbeddingLayer = InputEmbeddingLayer(W_EMB_SIZE, w_emb_weights, W_VOCAB_SIZE, W_SPECIAL_TOKEN,
                                              C_EMB_SIZE, C_VOCAB_SIZE)
    # 2. Embedding encoder block
    embedding_encoder = EncoderLayer(
        embedding_size=W_EMB_SIZE + C_EMB_SIZE,
        d_model=128,
        kernel_size=7,
        n_conv_layers=4,
        n_heads=8,
        survival_prob=1.0,
        l2_value=3e-7,
        n_blocks=1)

    # 3. Context-query attention
    context_query_attention = ContextQueryAttentionLayer(128)

    # 4. Model encoder layer
    model_encoder = EncoderLayer(
        embedding_size=512,
        d_model=128,
        kernel_size=7,
        n_conv_layers=2,
        n_heads=8,
        survival_prob=1.0,
        l2_value=3e-7,
        n_blocks=7)

    # 5. Output layer
    model_output = OutputLayer()

    context_embedded = inputEmbeddingLayer([w_context, c_context])
    context_encoded = embedding_encoder(context_embedded, training=training, mask=context_word_mask)

    query_embedded = inputEmbeddingLayer([w_query, c_query])
    query_encoded = embedding_encoder(query_embedded, training=training, mask=query_word_mask)

    attention_output = context_query_attention([context_encoded, query_encoded], [context_word_mask, query_word_mask])

    conv_layer_params = {
        "filters": 128,
        "kernel_size": 1,
        "padding": "same",  # necessary for residual blocks
        "data_format": "channels_last",
    }
    conv_1d = layers.SeparableConv1D(**conv_layer_params)
    attention_output = conv_1d(attention_output)

    m0 = model_encoder(attention_output, training=training, mask=context_word_mask)
    m1 = model_encoder(m0, training=training, mask=context_word_mask)
    m2 = model_encoder(m1, training=training, mask=context_word_mask)

    output = model_output([m0, m1, m2], mask=context_word_mask)

    # model = tf.keras.Model(
    #     inputs=[w_context, c_context],
    #     outputs=[output]
    # )

    print()


def loss_function(y_true, y_pred):
    # y_true = (batch_size, 2, 1) or (batch_size, 2)
    # y_pred = (batch_size, 2, n_words)

    assert int(tf.shape(y_true)[1]) == 2
    assert int(tf.shape(y_pred)[1]) == 2

    batch_size = int(tf.shape(y_true)[0])

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


test_model()

'''
#Test
true = tf.constant([[[0],[3]],[[1],[2]]])
pred = tf.constant([[[0.1,0.2,0.3,0.4],[0.2,0.35,0.65,0.78]],[[0.2,0.3,0.41,0.12],[0.13,0.45,0.78,0.21]]])

loss = loss_function(true,pred)
print(loss)
'''
