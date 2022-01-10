import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, final_learning_rate, warmup_steps):
        super(CustomSchedule, self).__init__()

        self.final_learning_rate = final_learning_rate

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # log_function = tf.math.log(tf.cast(step, tf.float32) + 1)
        # normalized_log_function = log_function / tf.math.log(self.warmup_steps)
        # scaled_normalized_log_function = normalized_log_function * self.final_learning_rate

        return tf.math.minimum(self.final_learning_rate,
                               0.001 / tf.math.log(999.) * tf.math.log(tf.cast(step, tf.dtypes.float32) + 1))

        # return tf.math.minimum(self.final_learning_rate, scaled_normalized_log_function)
