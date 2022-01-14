import tensorflow as tf
from tensorflow.keras import layers

from metrics import F1Score, EMScore, qa_loss
from context_query_attention import ContextQueryAttentionLayer
from encoding.encoder import EncoderLayer
from input_embedding.input_embedding_layer import InputEmbeddingLayer
from model_output import OutputLayer
import Config

# TODO: pass the correct word vocab size and ingore tokens
# f1_score = F1Score(vocab_size=Config.WORD_VOCAB_SIZE+1, ignore_tokens=tf.constant([[0], [1], [9], [10]]))
# em_score = EMScore(vocab_size=Config.WORD_VOCAB_SIZE+1, ignore_tokens=tf.constant([[0], [1], [9], [10]]))


class QACNNnet(tf.keras.Model):

    def __init__(self,
                 input_embedding_params,
                 embedding_encoder_params,
                 conv_query_attention_to_encoders_params,
                 model_encoder_params,
                 context_query_attention_params,
                 vocab_size,
                 ignore_tokens,
                 dropout_rate):

        super(QACNNnet, self).__init__()

        self.embedding = InputEmbeddingLayer(**input_embedding_params)
        self.embedding_encoder = EncoderLayer(**embedding_encoder_params)
        self.context_query_attention = ContextQueryAttentionLayer(**context_query_attention_params)
        self.model_encoder = EncoderLayer(**model_encoder_params)
        self.conv_1d = layers.SeparableConv1D(**conv_query_attention_to_encoders_params)
        self.model_output = OutputLayer()
        self.f1_score = F1Score(vocab_size=vocab_size, ignore_tokens=ignore_tokens)
        self.em_score = EMScore(vocab_size=vocab_size, ignore_tokens=ignore_tokens)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        self.dropout_rate = dropout_rate
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        self.model_is_training = None
        self.unaveraged_weights = None

    def call(self, inputs, training=None):

        assert len(inputs) == 4

        # Restore unaveraged weights
        if self.model_is_training == False and training == True:
            if self.unaveraged_weights is not None:
                # tf.print('restoring UNAVERAGED weights')
                for idx, var in enumerate(self.trainable_variables):
                    var.assign(tf.identity(self.unaveraged_weights[idx]))
                self.unaveraged_weights = None
        elif self.model_is_training == True and training == False:
            # tf.print('setting AVERAGED weights')
            # Save unaveraged weights and set the averaged ones
            self.unaveraged_weights = []
            for var in self.trainable_variables:
                # Deep copy the original variable
                self.unaveraged_weights.append(tf.identity(var))

                # Average the current variable
                var.assign(self.ema.average(var))

        self.model_is_training = training

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
        attention_output = self.conv_1d(attention_output)  # Used for obtaining back the expected number of channels

        # 4. Model encoder blocks
        m0 = self.model_encoder(attention_output, training=training, mask=context_mask)
        m1 = self.model_encoder(m0, training=training, mask=context_mask)

        # Apply dropout after 2 blocks
        # (Created here, and not in the init, for avoiding to see the dropout layer in the summary)
        m1 = tf.keras.layers.Dropout(self.dropout_rate)(m1)

        m2 = self.model_encoder(m1, training=training, mask=context_mask)

        # 5. Output block
        output = self.model_output([m0, m1, m2], mask=context_mask)

        return output

    def train_step(self, data):

        # Restore unaveraged weights
        # if self.model_is_training == False:
        #     if self.unaveraged_weights is not None:
        #         for idx, var in enumerate(self.trainable_variables):
        #             var.assign(tf.identity(self.unaveraged_weights[idx]))
        #         self.unaveraged_weights = None
        # self.model_is_training = True

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = qa_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Apply EMA
        self.ema.apply(self.trainable_variables)

        # Update the metrics
        self.loss_tracker.update_state(loss)

        self.f1_score.set_words_context(x[0])
        self.f1_score.update_state(y, y_pred)

        self.em_score.set_words_context(x[0])
        self.em_score.update_state(y, y_pred)


        # self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):

        # Save unaveraged weights and set the averaged ones
        # if self.model_is_training == True:
        #     self.unaveraged_weights = []
        #     for var in self.trainable_variables:
        #         # Deep copy the original variable
        #         self.unaveraged_weights.append(tf.identity(var))
        #
        #         # Average the current variable
        #         var.assign(self.ema.average(var))
        #
        #     # self.unaveraged_weights = self.trainable_variables
        #
        # self.model_is_training = False
            
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        loss = qa_loss(y, y_pred)
        #loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Update the metrics.
        self.loss_tracker.update_state(loss)

        self.f1_score.set_words_context(x[0])
        self.f1_score.update_state(y, y_pred)

        self.em_score.set_words_context(x[0])
        self.em_score.update_state(y, y_pred)

        # self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        # List our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker, self.f1_score, self.em_score]

    def swap_weights(self):
        """Swap the average and moving weights.
        This is a convenience method to allow one to evaluate the averaged weights
        at test time. Loads the weights stored in `self._average_weights` into the model,
        keeping a copy of the original model weights. Swapping twice will return
        the original weights.
        """
        if tf.distribute.in_cross_replica_context():
            strategy = tf.distribute.get_strategy()
            return strategy.run(self._swap_weights, args=())
        else:
            raise ValueError(
                "Swapping weights must occur under a " "tf.distribute.Strategy"
            )

    @tf.function
    def _swap_weights(self):
        def fn_0(a, b):
            return a.assign_add(b, use_locking=self._use_locking)

        def fn_1(b, a):
            return b.assign(a - b, use_locking=self._use_locking)

        def fn_2(a, b):
            return a.assign_sub(b, use_locking=self._use_locking)

        def swap(strategy, a, b):
            """Swap `a` and `b` and mirror to all devices."""
            for a_element, b_element in zip(a, b):
                strategy.extended.update(
                    a_element, fn_0, args=(b_element,)
                )  # a = a + b
                strategy.extended.update(
                    b_element, fn_1, args=(a_element,)
                )  # b = a - b
                strategy.extended.update(
                    a_element, fn_2, args=(b_element,)
                )  # a = a - b

        ctx = tf.distribute.get_replica_context()
        return ctx.merge_call(swap, args=(self._average_weights, self._model_weights))
