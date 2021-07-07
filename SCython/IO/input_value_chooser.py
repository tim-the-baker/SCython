import numpy as np
from SCython.SNG import SNG

def choose_input_values(num_values, bipolar, mode, seed=None, precision=None, data=None, start=None):
    """
    Method for choosing input values for simulation of a stochastic circuit.
    :param num_values: number of input values.
    :param bipolar: whether the input values are bipolar or not.
    :param mode: determines how input values are selected. Choices include:
    'rand' (or 0): values are chosen uniformly randomly.
    'data' (or 1) values are taken from a random subsequence of given data. requires use of data param.
    :param seed: optional parameter. If given, seed is passed to np.random.seed before executing the rest of this method.
    This parameter is used so that results can be replicated.
    :param precision: optional parameter. If given, the sampled input values will be quantized using SNG.q_floor.
    :param data: required parameter when mode = 'data' (or 1). This is the data from which a random sequence will be
    sampled from.
    :param start: optional parameter when mode = 'data' (or 1). If specified, start will used as the beginning of the
    sampled sequence rather than a random start value.
    :return:
    """
    if seed is not None:
        np.random.seed(seed)
    if mode == 0 or mode == 'rand':
        input_values = np.random.rand(num_values)
        if bipolar:
            input_values = 2*input_values - 1

    elif mode == 1 or mode == 'data':
        # check the required data parameter

        assert data is not None, "Error: Using choose_input_values method with mode = 'data', but no data given"
        if bipolar:
            assert -1 <= data <= 1, "Error: values in data given to choose_input values do not fall in [-1,1]"
        else:
            assert 0 <= data <= 1, "Error: values in data given to choose_input values do not fall in [0,1]"

        # Pick a random subset of the data of length "num"
        if start is None:
            start = np.random.randint(len(data) - num_values)
        input_values = data[start:start+num_values]

    else:
        raise ValueError

    if precision is not None:  # quantize the inputs if necessary
        input_values = SNG.q_floor(input_values, precision, signed=bipolar)

    return input_values


def choose_mux_weights(num_weights, mode, seed=None, data=None):
    """
    Method for choosing mux input weights for the simulation of a mux adder.
    :param num_weights: number of weights (should match the mux adder's input size)
    :param mode: determines how the weights are selected. Options include:
    'uniform' (or 0): every coefficient is 1/M
    'uniform signed' (or 1): every coefficient is randomly set to +/- 1/M
    'random' (or 2): weights are random over interval [0, 1]
    'random signed' (or 3): weights are random over interval [-1, 1]
    'random norm' (or 4): weights are random over [0, 1] but then normalized to 1.
    'random norm signed' (or 5): weights are random over [-1, 1] but then normalized to 1.
    'data' (or 6): weights are loaded from data. requires use of data parameter.
    :param seed: optional parameter. If given, seed is passed to np.random.seed before executing the rest of this method.
    This parameter is used so that results can be replicated.
    :param data: required parameter when mode = 'data' (or 6). This is the data from which weights will be loaded from.
    :return:
    """
    if seed is not None:
        np.random.seed(seed)

    if mode == 'uniform' or mode == 0:
        weights = np.ones(num_weights)/num_weights
    elif mode == 'uniform signed' or mode == 1:
        weights = ((-1) ** np.random.randint(0, 2, num_weights))/num_weights
    elif mode == 'rand' or mode == 2:
        weights = np.random.rand(num_weights)
    elif mode == 'rand signed' or mode == 3:
        weights = 2*np.random.rand(num_weights) - 1
    elif mode == 'rand norm' or mode == 4:
        weights = np.random.rand(num_weights)
        weights = weights / np.sum(weights)
    elif mode == 'rand norm signed' or mode == 5:
        weights = 2 * np.random.rand(num_weights) - 1
        weights = weights / np.sum(weights)
    elif mode == 'data' or mode == 6:
        raise NotImplementedError
    else:
        raise NotImplementedError

    return weights
