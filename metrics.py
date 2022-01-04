import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.test_message = 'None'

    def update_state(self, y_true, y_pred, sample_weight=None):
        print(f'update_state: {self.test_message}')
        tf.print(f'update_state: {self.test_message}')

    def set_message(self, message):
        self.test_message = message

    def result(self):
        return 2