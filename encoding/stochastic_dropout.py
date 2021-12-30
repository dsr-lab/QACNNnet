import random

def get_layer_survival_probability(n_layers: int,
                                  current_layer: int,
                                  survival_prob: float) ->float:

    '''
    Computes the probability of a layer to survive the stochastic dropout,
    using the official paper's formula.

    Parameters:
    -----------
    n_layers: int
        Number of sublayers inside the block
    current_layer: int
        Index of the current layer inside the block
    survival_prob: float
        Formula's parameter to compute the survival probability

    Returns:
    --------
    l_survival_prob: float
        probability of a layer to survive the stochastic dropout
    '''

    l_fract = current_layer / n_layers
    l_survival_prob = 1 - l_fract * (1 - survival_prob)
    return l_survival_prob

def keep_layer(n_layers: int,
              current_layer: int,
              survival_prob: float) ->bool:

    '''
    Decides whether a layer should be kept or not due to stochastic dropout.

    Parameters:
    -----------
    n_layers: int
        Number of sublayers inside the block
    current_layer: int
        Index of the current layer inside the block
    survival_prob: float
        Formula's parameter to compute the survival probability

    Returns:
    --------
    True if layer survives the stochastic dropout or not.
    '''

    prob = get_layer_survival_probability(n_layers, current_layer, survival_prob)
    event = random.random()
    return event < prob
