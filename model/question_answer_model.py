import tensorflow as tf
from tensorflow.keras import layers

from model.metrics import F1Score, EMScore, qa_loss
from layer_context_query_attention.context_query_attention import ContextQueryAttentionLayer
from layer_encoder.encoder import EncoderLayer
from layer_input_embedding.input_embedding_layer import InputEmbeddingLayer
from layer_model_output.model_output import OutputLayer
from utils import assert_tensor_validity


class QACNNnet(tf.keras.Model):

    def __init__(self,
                 input_embedding_params,
                 embedding_encoder_params,
                 conv_input_projection_params,
                 model_encoder_params,
                 context_query_attention_params,
                 output_params,
                 vocab_size,
                 ignore_tokens,
                 dropout_rate):
        '''
        Create the model that contains all the custom layers that compose the QACNNet.

        Parameters:
        -----------
        input_embedding_params: dict
            Dictionary containing all the parameters required for configuring the InputEmbeddingLayer
            (1st network block)
        embedding_encoder_params: dict
            Dictionary containing all the parameters required for configuring the Embedding Encoder
            (2nd network block)
        conv_input_projection_params: dict
            Dictionary containing the parameters for a 1x1 convolution used for resizing the number
            of dimensions to D_MODEL (see the config file)
        model_encoder_params: dict
            Dictionary containing all the parameters required for configuring the Model Encoder
            (3rd network block)
        context_query_attention_params: dict
            Dictionary containing the parameters required for the ContextQueryAttention layer
            (4th network block)
        output_params: dict
            Dictionary containing the parameters required for the Output layer
            (5th network block. This is the last network block)
        vocab_size: int
            The number of words in the vocaboulary
        c_vocab_size: int
            The number of characters in the vocaboulary
        ignore_tokens: tf.tensor
            Tensor containing an array of tokens that must be ignored when computing the model metrics
        dropout_rate: float
            The dropout rate.
            Passing 0.0 means that dropout is not applied.
        '''

        super(QACNNnet, self).__init__()

        self.dropout_rate = dropout_rate

        # Layers creation
        self.embedding = InputEmbeddingLayer(**input_embedding_params)
        self.embedding_encoder = EncoderLayer(**embedding_encoder_params)

        self.context_query_attention = ContextQueryAttentionLayer(**context_query_attention_params)

        self.model_encoder = EncoderLayer(**model_encoder_params)
        self.conv_1d = layers.SeparableConv1D(**conv_input_projection_params)
        self.model_output = OutputLayer(**output_params)

        # Custom metrics and loss
        self.f1_score = F1Score(vocab_size=vocab_size, ignore_tokens=ignore_tokens)
        self.em_score = EMScore(vocab_size=vocab_size, ignore_tokens=ignore_tokens)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

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

        assert_tensor_validity(context_encoded, "context_encoded")
        assert_tensor_validity(query_encoded, "query_encoded")

        # 3. Context-query attention block
        attention_output = self.context_query_attention([context_encoded, query_encoded], [context_mask, query_mask])
        assert_tensor_validity(attention_output, "attention_output1")
        attention_output = self.conv_1d(attention_output)  # Used for obtaining back the expected number of channels
        assert_tensor_validity(attention_output, "attention_output2")
        attention_output = tf.keras.layers.Dropout(self.dropout_rate)(attention_output)
        assert_tensor_validity(attention_output, "attention_output3")



        # 4. Model encoder blocks
        m0 = self.model_encoder(attention_output, training=training, mask=context_mask)
        assert_tensor_validity(m0, "m0")
        m1 = self.model_encoder(m0, training=training, mask=context_mask)
        assert_tensor_validity(m1, "m1")

        # Apply dropout after 2 blocks
        # (Created here, and not in the init, for avoiding to see the dropout layer in the model summary)
        m1 = tf.keras.layers.Dropout(self.dropout_rate)(m1)
        assert_tensor_validity(m1, "m1dropout")

        m2 = self.model_encoder(m1, training=training, mask=context_mask)
        assert_tensor_validity(m1, "m2")

        # 5. Output block
        output = self.model_output([m0, m1, m2], mask=context_mask)
        assert_tensor_validity(output, "output")

        return output

    def train_step(self, data):

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = qa_loss(y, y_pred)

            loss += sum(self.losses)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

            assert_tensor_validity(loss, "loss")
            assert_tensor_validity(scaled_loss, "scaled_loss")

        # Compute gradients
        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

        # Gradients clipping
        capped_grads, _ = tf.clip_by_global_norm(gradients, 5.0)

        # assert_tensor_validity(scaled_gradients, "scaled_gradients")
        # assert_tensor_validity(gradients, "gradients")
        # assert_tensor_validity(capped_grads, "capped_grads")

        # Update weights
        self.optimizer.apply_gradients(zip(capped_grads, trainable_vars))
        #self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        self.loss_tracker.update_state(loss)

        self.f1_score.set_words_context(x[0])
        self.f1_score.update_state(y, y_pred)

        self.em_score.set_words_context(x[0])
        self.em_score.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):

        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        loss = qa_loss(y, y_pred)
        # loss += sum(self.losses)  # Not required to add regularization losses during the testing phase.

        # Update the metrics.
        self.loss_tracker.update_state(loss)

        self.f1_score.set_words_context(x[0])
        self.f1_score.update_state(y, y_pred)

        self.em_score.set_words_context(x[0])
        self.em_score.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # List our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker, self.f1_score, self.em_score]
