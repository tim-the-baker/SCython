# TODO docstrings
import numpy as np
import math
import logging

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

    def gen_selects(self, SN_length):
        """
        Abstract method for generating the mux select inputs based on SN length
        :param int SN_length:
        :return:
        """

        raise NotImplementedError

    def gen_verilog(self):
        raise NotImplementedError


class Counter_MSG(MSG):
    def __init__(self, precision, tree_height, weights, max_val: int = None):
        """
        :param precision:
        :param tree_height:
        :param weights:
        :param max_val:
        """
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
