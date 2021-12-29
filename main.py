import tensorflow as tf
import numpy as np

from input_embedding.input_embedding_layer import InputEmbeddingLayer


def test_input_embedding():
    N_WORDS = 30
    N_CHAR = 16
    W_EMB_SIZE = 300
    W_VOCAB_SIZE = 100
    W_SPECIAL_TOKENS = 1

    C_EMB_SIZE = 200
    C_VOCAB_SIZE = 50
    C_CONV_OUTPUT_SIZE = C_EMB_SIZE
    C_CONV_KERNEL_SIZE = 5

    input_embedding_layer = InputEmbeddingLayer(
        w_emb_size=W_EMB_SIZE,
        w_pretrained_weights=np.random.rand(W_VOCAB_SIZE, W_EMB_SIZE).astype(np.float32),
        w_vocab_size=W_VOCAB_SIZE,
        w_n_special_tokens=W_SPECIAL_TOKENS,
        c_emb_size=C_EMB_SIZE,
        c_vocab_size=C_VOCAB_SIZE,
        c_conv_output_size=C_CONV_OUTPUT_SIZE,  # 96 in the blog code, 128 in the paper
        c_conv_kernel_size=C_CONV_KERNEL_SIZE,  # 5 in the blog code, not clear in the paper (maybe 7??)
    )

    input1 = tf.keras.Input(shape=(N_WORDS), name="input_words")
    input2 = tf.keras.Input(shape=(N_WORDS, N_CHAR), name="input_chars")
    embedded_input = input_embedding_layer((input1, input2))
    m = tf.keras.Model(inputs=[input1, input2], outputs=embedded_input)
    m.summary()

    tf.keras.utils.plot_model(m, "sample_model.png", show_shapes=True, expand_nested=True)


def main():
    print('main function')
    # test_input_embedding()
    questions = np.random.rand(32, 30, 128)
    contexts = np.random.rand(32, 400, 128)
    #w = np.random.rand(32, 400)

    #c = np.dot(np.transpose(questions), w)
    #print(c.shape)

    #d = np.dot(c, contexts)
    #print(d.shape)

    q = np.random.rand(32, 30, 1)
    t = np.random.rand(32, 400, 1)
    dot = np.dot(q, t)
    b = tf.concat((q,t), axis=1)
    print(b.shape)


    print()


if __name__ == '__main__':
    main()
