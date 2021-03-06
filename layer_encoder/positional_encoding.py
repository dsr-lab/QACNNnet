import numpy as np
import tensorflow as tf

# This module contains those methods that allow to compute vectors for positional layer_encoder.

def get_angles(pos, i, d_model):
    '''
    Apply official standard positional layer_encoder formula to get positional
    angles of a series of positions in a given number of dimensions.

    Parameters:
    -----------
    pos: int
        Position of a word or sequence of positions of words.
    i: int
        Index of a dimension of the embedding or sequence of indices of all the dimensions of the embedding.
    d_model: int
        Number of dimensions of the embedding.

    Returns:
    --------
    Angles associated with each position.
    '''

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def get_encoding(length, d_model):
    '''
    Get the positional layer_encoder of a set of positions.

    Parameters:
    ----------
    length: int
        Number of positions to encode (from 0 to length-1)
    d_model: int
        Number of dimensions used (usually embedding size).

    Returns:
    --------
    pos_encoding: tf.Tensor
        Positional layer_encoder of the positions.
    '''

    # Compute the angles of each position
    angle_rads = get_angles(np.arange(length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Compute the sin of angles of even positions
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Compute the cosine of angles of odd positions
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Get positional layer_encoder for each position
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=np.float32)
