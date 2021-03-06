import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, final_learning_rate, warmup_steps):
        '''
        Create custom schedule that is used for applying a learning rate warmup

        Parameters:
        -----------
        final_learning_rate: float
            The final learning rate value that will be reached after the warmup phase
        warmup_steps: float
            The number of steps that must be execute before reaching the final_learning_rate
        '''

        super(CustomSchedule, self).__init__()

        self.final_learning_rate = final_learning_rate

        self.warmup_steps = warmup_steps

    def __call__(self, step):

        # Compute the learning rate
        log_function = tf.math.log(tf.cast(step, tf.float32) + 1)
        normalized_log_function = log_function / tf.math.log(self.warmup_steps)
        scaled_normalized_log_function = normalized_log_function * self.final_learning_rate

        return tf.math.minimum(self.final_learning_rate, scaled_normalized_log_function)
