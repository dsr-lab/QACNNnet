import tensorflow as tf
import numpy as np


def main():
    print('main function')
        
    a = np.asarray(
        [
            [
                [
                    [1, 2, 3, 66, 5],
                    [1, 2, 32, 4, 5],
                    [1, 100, 3, 4, 5],
                    [99, 2, 3, 4, 58]
                ],
                [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5]
                ],
            ]
        ]

    )
    b = a.shape
    c = tf.math.reduce_max(a, axis=2)

    BATCH_SIZE = 2
    N_WORDS = 2
    N_CHAR = 6
    N_FILTER = 5
    EMB_SIZE = 20
    CONV_SIZE = 3
    VOCAB_SIZE = 100

    sentence_input = np.random.randint(VOCAB_SIZE, size=(BATCH_SIZE, N_WORDS, N_CHAR))

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

    y2 = tf.reduce_max(y2, axis=1)
    y2 = tf.reshape(y2, [BATCH_SIZE, N_WORDS, N_FILTER])

    if (y.numpy() == y2.numpy()).all():
        print("equal!")

    print()


if __name__ == '__main__':
    main()
