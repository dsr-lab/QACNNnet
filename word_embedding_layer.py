import numpy as np
import tensorflow as tf


class WordEmbeddingLayer(tf.keras.layers.Layer):

    # input-independent initialization
    def __init__(self, emb_size, emb_weight_matrix, vocab_size, n_special_tokens):
        super(WordEmbeddingLayer, self).__init__()

        # Set variables
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.n_special_tokens = n_special_tokens

        # Placeholders (Defined later).
        self.emb_layer = None
        self.special_emb_layer = None

        self.emb_layer_weights, self.special_emb_layer_weights = \
            self._set_weights(emb_weight_matrix, emb_size, n_special_tokens)

    # input-dependent initialization
    def build(self, input_lenght):
        self.emb_layer = tf.keras.layers.Embedding(
            self.vocab_size + 1,  # +1 for padding
            self.emb_size,
            input_length=input_lenght[-1],
            trainable=False,
            mask_zero=True,
            weights=[self.emb_layer_weights],
        )

        self.special_emb_layer = tf.keras.layers.Embedding(
            self.n_special_tokens + 1,  # +1 for padding
            self.emb_size,
            input_length=input_lenght[-1],
            trainable=True,
            mask_zero=True,
            weights=[self.special_emb_layer_weights]
        )

    # forward computation
    def call(self, inputs):
        # Tranform into padding all the tokens that are not considered as special (e.g., <UNK>)
        n_valid_tokens = self.vocab_size - self.n_special_tokens
        special_tokens_input = tf.keras.layers.Lambda(lambda x: x - n_valid_tokens)(inputs)
        special_tokens_input = tf.keras.layers.Activation('relu')(special_tokens_input)

        # Apply the embedding using both layers
        embedded_sequences = self.emb_layer(inputs)
        embedded_special = self.special_emb_layer(special_tokens_input)

        # Add the matrices to obtain a single embedding result
        embedded_sequences = tf.keras.layers.Add()([embedded_sequences, embedded_special])

        return embedded_sequences

    @staticmethod
    def _set_weights(pretrained_weights, emb_size, n_special_tokens):
        zero_val = np.zeros((1, emb_size))

        emb_layer_weights = pretrained_weights[:-n_special_tokens]
        emb_layer_weights = np.insert(emb_layer_weights, 0, zero_val, axis=0)  # Add padding

        for i in range(n_special_tokens):
            emb_layer_weights = np.append(emb_layer_weights, zero_val, axis=0)

        special_emb_layer_weights = pretrained_weights[-n_special_tokens:]
        special_emb_layer_weights = np.insert(special_emb_layer_weights, 0, zero_val, axis=0)  # Add padding

        return emb_layer_weights, special_emb_layer_weights
