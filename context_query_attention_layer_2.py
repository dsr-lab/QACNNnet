import tensorflow as tf

import Config


class ContextQueryAttentionLayer2(tf.keras.layers.Layer):
    def __init__(self, n_channels=Config.D_MODEL, max_context_words=Config.MAX_CONTEXT_WORDS,
                 max_query_words=Config.MAX_QUERY_WORDS, dropout_rate=Config.DROPOUT_RATE, l2_rate=Config.L2_RATE):
        super(ContextQueryAttentionLayer2, self).__init__()



        self.max_context_words = max_context_words
        self.max_query_words = max_query_words

        l2 = None if l2_rate == 0.0 else tf.keras.regularizers.l2(l2_rate)

        self.weights4arg0 = self.add_weight(
            shape=(n_channels, 1), initializer='glorot_uniform', trainable=True, regularizer=l2
        )

        self.weights4arg1 = self.add_weight(
            shape=(n_channels, 1), initializer='glorot_uniform', trainable=True, regularizer=l2
        )

        self.weights4mlu = self.add_weight(
            shape=(1, 1, n_channels), initializer='glorot_uniform', trainable=True, regularizer=l2
        )
        self.biases = self.add_weight(
            shape=[self.max_query_words], initializer='zeros', trainable=True, regularizer=l2
        )

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, masks):

        c = inputs[0]
        c = self.dropout(c)
        q = inputs[1]
        q = self.dropout(q)

        c_mask = masks[0]
        q_mask = masks[1]

        subres0 = tf.tile(tf.tensordot(c, self.weights4arg0, axes=(2, 0)), [1, 1, self.max_query_words])
        subres1 = tf.tile(tf.transpose(tf.tensordot(q, self.weights4arg1, axes=(2, 0)), perm=(0, 2, 1)),
                          [1, self.max_context_words, 1])
        subres2 = batch_dot(c * self.weights4mlu, tf.transpose(q, perm=(0, 2, 1)))

        res = subres0 + subres1 + subres2

        S = tf.nn.bias_add(res, self.biases)

        mask_q = tf.expand_dims(q_mask, 1)
        S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
        mask_c = tf.expand_dims(c_mask, 2)
        S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), axis=1), (0, 2, 1))
        c2q = tf.matmul(S_, q)
        q2c = tf.matmul(tf.matmul(S_, S_T), c)
        attention_outputs = [c, c2q, c * c2q, c * q2c]

        return tf.concat(attention_outputs, axis=-1)


def mask_logits(inputs, mask, mask_value=-1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def ndim(x):
    """Copied from keras==2.0.6
    Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out
