import numpy as np
import tensorflow as tf

def get_angles(pos: int,
              i: int,
              d_model: int) ->np.ndarray:

    '''
    Apply official standard positional encoding formula to get positional
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

    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def get_encoding(length: int,
                d_model: int) ->tf.Tensor:

    '''
    Get the positional encoding of a set of positions.

    Parameters:
    ----------
    length: int
        Number of positions to encode (from 0 to length-1)
    d_model: int
        Number of dimensions used (usually embedding size).

    Returns:
    --------
    pos_encoding: tf.Tensor
        Positional encoding of the positions.
    '''

    #Compute the angles of each position
    angle_rads = get_angles(np.arange(length)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    #Compute the sin of angles of even positions
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    #Compute the cosin of angles of odd positions
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    #Get positional encoding for each position
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
