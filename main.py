import tensorflow as tf
import numpy as np


# from tensorflow.keras import layers


def main():
    print('main function')
    embedding_layer_weights = np.zeros((2, 5))
    print(embedding_layer_weights.shape)
    a = np.ones((1,5))
    embedding_layer_weights = np.append(embedding_layer_weights, a, axis=0)
    print(embedding_layer_weights.shape)
    print(embedding_layer_weights)


if __name__ == '__main__':
    main()
