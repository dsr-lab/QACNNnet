import tensorflow as tf
from tensorflow.keras import layers

from metrics import F1Score
from context_query_attention import ContextQueryAttentionLayer
from encoding.encoder import EncoderLayer
from input_embedding.input_embedding_layer import InputEmbeddingLayer
from model_output import OutputLayer


class QACNNnet(tf.keras.Model):

    def __init__(self, input_embedding_params, embedding_encoder_params, conv_layer_params, model_encoder_params):
        super(QACNNnet, self).__init__()

        self.embedding = InputEmbeddingLayer(**input_embedding_params)
        self.embedding_encoder = EncoderLayer(**embedding_encoder_params)
        self.context_query_attention = ContextQueryAttentionLayer()
        self.model_encoder = EncoderLayer(**model_encoder_params)
        self.conv_1d = layers.SeparableConv1D(**conv_layer_params)
        self.model_output = OutputLayer()

        self.unwanted_tokens = []  # TODO: this should contain

    def call(self, inputs, training=None):
        assert len(inputs) == 4

        words_context = inputs[0]
        characters_context = inputs[1]
        words_query = inputs[2]
        characters_query = inputs[3]

        # 1. Embedding blocks
        context_embedded, context_mask = self.embedding([words_context, characters_context])
        query_embedded, query_mask = self.embedding([words_query, characters_query])

        # 2. Embedding encoder blocks
        context_encoded = self.embedding_encoder(context_embedded, training=training, mask=context_mask)
        query_encoded = self.embedding_encoder(query_embedded, training=training, mask=query_mask)

        # 3. Context-query attention block
        attention_output = self.context_query_attention([context_encoded, query_encoded], [context_mask, query_mask])
        attention_output = self.conv_1d(attention_output)

        # 4. Model encoder blocks
        m0 = self.model_encoder(attention_output, training=training, mask=context_mask)
        m1 = self.model_encoder(m0, training=training, mask=context_mask)
        m2 = self.model_encoder(m1, training=training, mask=context_mask)

        # 5. Output block
        output = self.model_output([m0, m1, m2], mask=context_mask)

        return output

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # TODO: pass to the metric all the necessary for properly computing the f1 score
        if len(self.metrics) > 1:
            metric = self.metrics[1]
            if isinstance(metric, F1Score):
                metric.set_words_context(x[0])

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
