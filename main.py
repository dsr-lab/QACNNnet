import tensorflow as tf
import numpy as np

from input_embedding.input_embedding_layer import InputEmbeddingLayer


def main():
    print('main function')

    BATCH_SIZE = 2
    N_WORDS = 2
    N_CHAR = 6
    N_FILTER = 96  # blog=96
    EMB_SIZE = 20
    CONV_SIZE = 3  # blog=5
    VOCAB_SIZE = 100

    c_sentence_input = np.random.randint(50, size=(BATCH_SIZE, N_WORDS, N_CHAR))
    w_sentences_input = np.random.randint(100, size=(BATCH_SIZE, N_WORDS))


    input_embedding_layer = InputEmbeddingLayer(
        w_emb_size=20,
        w_pretrained_weights=np.random.rand(100, 20).astype(np.float32),
        w_vocab_size=100,
        w_n_special_tokens=1,
        c_emb_size=20,
        c_vocab_size=50,
        c_conv_output_size=5)

    embedded_input = input_embedding_layer(w_sentences_input, c_sentence_input)
    embedded_input_shape = embedded_input.shape

    # s3 = tf.concat([s1, s2], axis=2)
    # s4 = tf.keras.layers.concatenate([s1, s2], axis=2)

    sentence_input = np.random.randint(50, size=(BATCH_SIZE, N_WORDS, N_CHAR))
    emb = tf.keras.layers.Embedding(VOCAB_SIZE, EMB_SIZE, input_length=6)
    x = emb(sentence_input)

    # input_shape=input_shape[2:]
    conv_layer = tf.keras.layers.Conv1D(N_FILTER, CONV_SIZE, activation='relu')
    y = conv_layer(x)
    # Conv output shape = (BATCH_SIZE, N_WORDS, CONV_SIZE, N_FILTERS)
    # reduce along the second axis (basically take the max
    y = tf.math.reduce_max(y, axis=2)

    x2 = tf.reshape(x, [BATCH_SIZE * N_WORDS, N_CHAR, EMB_SIZE])
    y2 = conv_layer(x2)

    # y2 = tf.reduce_max(y2, axis=1)
    y2 = tf.keras.layers.GlobalMaxPool1D()(y2)
    y2 = tf.reshape(y2, [BATCH_SIZE, N_WORDS, N_FILTER])

    if (y.numpy() == y2.numpy()).all():
        print("equal!")

    f = highway(y)

    print()


def highway(x, activation=None, num_layers=2, dropout=0.0):
    # size = kernel_number (last x dimension)
    n_filters = x.shape[-1]  # output channels equal to input channels
    kernel_size = 1

    for i in range(num_layers):
        # 96 in the paper (n_filters)
        T = tf.keras.layers.Conv1D(n_filters, kernel_size, activation='sigmoid')(x)

        H = tf.keras.layers.Conv1D(n_filters, kernel_size, activation=activation)(x)
        H = tf.nn.dropout(H, dropout)
        x = H * T + x * (1.0 - T)
    return x


if __name__ == '__main__':
    main()
