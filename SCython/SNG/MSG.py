# TODO docstrings
import numpy as np
import math
import logging
import SCython.Utilities.seq_utils as seq_utils

class MSG:
    """ Abstract class for mux select input generators (MSGs). """
    def __init__(self, precision, tree_height, weights):
        """
        :param int precision:
        :param int tree_height:
        :param numpy.ndarray weights:
        """
        self.n = precision
        self.m = tree_height
        self.weights = weights
        self.seq = None

    def gen_selects(self, SN_length):
        """
        Abstract method for generating the mux select inputs based on SN length
        :param int SN_length:
        :return:
        """

        raise NotImplementedError

    def gen_select_bit_streams(self):
        assert type(self) is Counter_MSG or type(self) is VDC_MSG,\
            "gen_select_bit_streams is not implemented for non deterministic MSGs yet."
        select_SNs = [[int(bit) for bit in reversed(np.binary_repr(number, self.n))] for number in self.seq[0]]
        return np.array(select_SNs).T


class Counter_MSG(MSG):
    def __init__(self, precision, tree_height, weights, max_val: int = None):
        super().__init__(precision, tree_height, weights)

        if precision > tree_height:
            logging.warning("Warning: Counter_MSG's precision cannot exceed the tree's height. Setting precision=height.")
            precision = tree_height

        self.max_val = int(2**precision) if max_val is None else max_val

        # The following "None" index transforms self.seq.shape from (max_val,) to (1, max_val)
        self.seq = np.arange(0, self.max_val)[None, :]

        assert self.max_val <= 2**precision

    def gen_selects(self, SN_length):
        num_repeats = math.ceil(SN_length / self.max_val)
        return self.seq.repeat(num_repeats)[:SN_length]


class VDC_MSG(MSG):
    def __init__(self, precision, tree_height, weights, max_val=None, vdc_seq=None):
        super().__init__(precision, tree_height, weights)

        if precision > tree_height:
            logging.warning("Warning: Counter_MSG's precision cannot exceed the tree's height. Setting precision=height.")
            precision = tree_height

        self.max_val = int(2**precision) if max_val is None else max_val
        assert self.max_val <= 2**precision

        self.seq = seq_utils.get_vdc(precision, verbose=True) if vdc_seq is None else vdc_seq
        self.seq = self.seq[None, :]

    def gen_selects(self, SN_length):
        num_repeats = math.ceil(SN_length / self.max_val)
        return self.seq.repeat(num_repeats)[:SN_length]


class HYPER_MSG(MSG):
    def __init__(self, precision, tree_height, weights, max_val=None, vdc_seq=None):
        super().__init__(precision, tree_height, weights)

        if precision > tree_height:
            logging.warning("Warning: Counter_MSG's precision cannot exceed the tree's height. Setting precision=height.")
            precision = tree_height

        self.max_val = int(2**precision) if max_val is None else max_val
        assert self.max_val <= 2**precision

        self.seq = None

    def gen_selects(self, SN_length):
        num_repeats = math.ceil(SN_length / self.max_val)
        seq = np.random.permutation(self.max_val)[None]
        return seq.repeat(num_repeats)[:SN_length]

    def gen_select_bit_streams(self):
        seq = np.random.permutation(self.max_val)
        select_SNs = [[int(bit) for bit in reversed(np.binary_repr(number, self.n))] for number in seq]
        return np.array(select_SNs).T
