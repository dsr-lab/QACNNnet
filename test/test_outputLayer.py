from unittest import TestCase

import numpy as np

from layer_model_output.model_output import OutputLayer


class TestOutputLayer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.BATCH_SIZE = 2
        cls.N_CONTEXT = 5
        cls.N_DIM = 4

    def test_call_verifyOutputShape(self):
        # Arrange
        layer = OutputLayer(l2_rate=3e-7)
        m0 = np.random.rand(self.BATCH_SIZE, self.N_CONTEXT, self.N_DIM).astype(np.float16)
        m1 = np.random.rand(self.BATCH_SIZE, self.N_CONTEXT, self.N_DIM).astype(np.float16)
        m2 = np.random.rand(self.BATCH_SIZE, self.N_CONTEXT, self.N_DIM).astype(np.float16)

        # Act
        result = layer([m0, m1, m2])
        expected_shape = (self.BATCH_SIZE, 2, self.N_CONTEXT)

        # Assert
        self.assertEqual(result.shape, expected_shape)
