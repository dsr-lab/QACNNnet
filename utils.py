import tensorflow as tf


def assert_tensor_validity(x, operation_name):

    # Check only for nan
    '''
    mask = tf.math.is_nan(x)
    mask = tf.math.reduce_any(mask)

    tf.debugging.Assert(
        tf.equal(mask, True), [x], summarize=None, name=f'CustomExceptionIsNan-{operation_name}'
    )
    '''

    # Check both for nan and inf
    mask = tf.math.is_finite(x)
    mask = tf.math.reduce_all(mask)

    tf.debugging.Assert(
        tf.equal(mask, True), [x], summarize=None, name=f'CustomExceptionIsFinite-{operation_name}'
    )