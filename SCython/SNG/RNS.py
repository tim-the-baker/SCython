import numpy as np
import SCython.IO.seq_utils as seq_utils
import math

class RNS:
    """ Abstract class for stochastic computing random number sources """
    def __init__(self, n, **kwargs):
        self._n: int = n
        self._max_N: int = -1
        self.name: str = ""
        self.legend: str = ""
        self.is_hardware: bool = False

    def _gen_RN(self, N, shape, share) -> np.ndarray:
        """
        Helper function for gen_SN. This method must be overridden in the subclass and should implement the RNS.
        See the gen_RN method for more info or view some of the including RNS subclasses for examples.
        :param int N: SN length
        :param tuple shape: dimensions of random number array.
        :param bool share: whether or not to share the RNS amongst all SNs.
        :return: the random number array
        """
        raise NotImplementedError

    def gen_RN(self, N: int, shape, share, inv_mask=None) -> np.ndarray:
        """
        Generates an array of random numbers used for SN generation. Can implement sharing the RNS directly by setting
        share=True and can implement sharing the inverted RNS by using share=True and the inv_mask param. This
        method implements the part of RN generation that applies to all RNS while the helper _gen_RN method implements
        each RNS specific manner of creating random numbers.
        :param int N: SN length.
        :param tuple shape: dimension of the random number array.
        :param bool share: whether or not to share the RNS amongst all RNs in the array.
        :param np.ndarray inv_mask: inv_mask is a Boolean array whose size should match the shape param. Each entry in
        inv_mask corresponds to a random number in the final random number array. When the inv_mask entry is True, the
        corresponding shared random number is inverted and when inv_mask entry is False, the corresponding shared random
        number is left unchanged.
        :return: the random number array whose shape is determined by the N and shape params: (*shape, N).
        """
        # figure out how many times the RNS needs to repeat itself.
        # E.g. a 4-bit LFSR RNS with sequence length 15 needs to repeat its sequence thrice when SN length, N, is 32.
        repeats = np.ceil(N / self._max_N)

        # generate random numbers according to the RNS being used.
        if repeats > 1:
            Rs = self._gen_RN(self._max_N, shape, share)
        else:
            Rs = self._gen_RN(N, shape, share)

        # if the RNS is shared, apply mask to necessary elements of R
        if share and inv_mask is not None:
            assert inv_mask.shape == shape, f"Mask shape: {inv_mask.shape} and given input shape: {shape} must match."
            # inverting all RNS bits is the same as doing Rs=(2^n -1-Rs)
            Rs[inv_mask] = int(2**self.n) - 1 - Rs[inv_mask]

        # repeat the RNS if necessary (see first comment for what it means for the RNS to repeat itself).
        if repeats > 1:
            Rs = np.tile(A=Rs, reps=repeats)[..., 0:N]

        return Rs

    def gen_verilog(self, IDs) -> str:
        """
        TODO: This docstring
        :param IDs:
        :return:
        This method generates Verilog code for the RNS. Software RNSs should return 'None' for this
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name

    def info(self) -> str:
        """
        returns a string with basic information about the RNS.
        :return:
        """
        return f"### RNS Info: name={self.name} n={self._n} max_N={self._max_N}"

    def verilog_info(self, verbose):
        """
        TODO: this docstring
        :param verbose:
        :return:
        """
        if not self.is_hardware:
            print(f"Warning: Trying to generate verilog for software RNS type {self.name}")
        elif verbose:
            print(f"### Generating verilog for RNS {self.name}")

    @property
    def n(self):
        return self._n

    @property
    def max_N(self):
        return self._max_N


# TODO: Update this class from PyTorch
'''
class Bernoulli_RNS(RNS):
    """
    Software implemented Bernoulli-type RNS. This RNS is mainly used to check theoretical derivations made using the
    Bernoulli model of SNs. It relies on NumPy's np.random.randint method whose RN quality is sufficiently high to
    imitate independent random numbers.
    """
    def __init__(self, n):
        """
        :param n: the precision, in bits, of the random numbers generated.
        """
        super().__init__(n)
        # software Bernoulli RNS can handle arbitrarily large SNs so max_N maintains its default value
        self._max_N = self._max_N

        self.name = "Bernoulli"
        self.legend = "Bern"  # string used in plot legends
        self.hardware = False

    def _gen_RN(self, N, shape, share):
        if share:
            R = np.random.randint(low=0, high=int(2**self.n), size=N)
            Rs = np.repeat(R[np.newaxis], repeats=math.prod(shape), axis=0).reshape(*shape, N)
        else:
            Rs = np.random.randint(low=0, high=16, size=(*shape, N))

        return Rs

    def gen_verilog(self, IDs=None, verbose=True):
        super().verilog_info(verbose)
        return None
'''

# TODO: Update this class from PyTorch
class Hypergeometric_RNS(RNS):
    # software hypergeometric RNG
    def __init__(self, n):
        super().__init__(n)
        self.name = "Hypergeometric"
        self.legend = "Hyper"
        self.is_hardware = False
        self._max_N = int(2**self.n)

    def _gen_RN(self, N, SN_array_shape, share_RNS):
        assert N <= self._max_N
        if share_RNS:
            shared_R = np.random.permutation(self._max_N)
            Rs = np.tile(shared_R, (*SN_array_shape, 1))
        else:
            Rs = np.array([np.random.permutation(self._max_N)[:N] for _ in range(math.prod(SN_array_shape))]).reshape((*SN_array_shape, N))

        return Rs

    def gen_verilog(self, verbose=True):
        super().verilog_info(verbose)
        return None


# TODO: Update this class from PyTorch
'''
class FSR_RNS(RNS):
    # hardware linear (or non-linear) feedback shift register RNG
    def __init__(self, n, **kwargs):
        super().__init__(n)

        self.nonlinear = kwargs.get("nonlinear")  # True if you want LFSR, False if you want NLFSR
        self.feedback = kwargs.get("feedback")  # FSR feedback type (internal, external or all)
        self.seq_idxs = kwargs.get("seq_idx")  # idx of seq from seqs to use
        if self.feedback is None:
            print("Warning: No LFSR feedback given, using 'e' for external feedback")
            self.feedback = 'e'

        # LFSR seqs to use for SN generation
        self.seqs = kwargs.get("seqs", seq_utils.get_LFSR_seqs(n, self.feedback, extended=self.nonlinear, verbose=True))

        self.name = "NLFSR" if self.nonlinear else "LFSR"
        self.legend = "NLFSR" if self.nonlinear else "LFSR"
        self.is_hardware = True
        self._max_N = int(2**n) if self.nonlinear else int(2**n) - 1

        assert (not self.nonlinear) or self.feedback == 'e', "NLFSRs are only implemented for external feedback"

    def _gen_RN(self, N, shape, share, starts=None, runs=1):
        assert N <= self._max_N
        pow2n = int(2 ** self._n)
        # Rs = torch.empty((runs, *shape, N))
        if share:
            assert self.seq_idxs is None or len(self.seq_idxs) > runs, \
                f"Only {len(self.seq_idxs)} seqs given, but are using {runs} runs"
            seq_idxs = torch.randint(len(self.seqs), (runs,)) if self.seq_idxs is None else self.seq_idxs
            if starts is None:
                # random starts
                start_idxs = torch.randint(N, (runs,))
            elif starts == -1:
                # 0 starts
                start_idxs = torch.zeros(runs)
            else:
                start_idxs = starts

            Rs = [torch.roll(self.seqs[seq_idxs[r]], (-start_idxs[r],)).repeat(*shape, 1) for r in range(runs)]
            Rs = torch.stack(Rs)  # convert list of tensors to tensor.

        else:
            assert self.seq_idxs is None, "LFSR no share for given seq_idxs not implemented"
            Rs = torch.empty((runs, shape.numel(), N))
            for r in range(runs):
                seq_idxs = torch.full((shape.numel(),), -1)
                start_idxs = torch.full((shape.numel(),), -1)
                # Picks seqs asserting that no two seqs that come from the same LFSR are within 2*n of each other index.
                for idx in range(shape.numel()):
                    done = False

                    seq_idx, start_idx = None, None
                    while not done:
                        seq_idx = torch.randint(len(self.seqs), (1,))
                        start_idx = torch.randint(pow2n - 1, (1,))

                        done = True
                        # if another LFSR is already uses this sequence, then this LFSR should start in a suitably far start state
                        if seq_idx in seq_idxs:
                            other_start_idxs = seq_idxs[seq_idxs == seq_idx]
                            if (torch.abs(other_start_idxs - start_idx) < 2 * self._n).any():
                                done = False

                    seq_idxs[idx] = seq_idx
                    start_idxs[idx] = start_idx
                    Rs[r, idx] = torch.roll(self.seqs[seq_idx], (-start_idx.item(),))

            Rs = Rs.reshape((runs, *shape, N))
        return Rs

    def gen_verilog(self, IDs, verbose=True):
        super().verilog_info(verbose)
        assert self.feedback == 'e', "Only implemented external feedback for simplicity!"
        name = "nlfsr" if self.nonlinear else "lfsr"
        feeds = seq_utils.get_LFSR_feeds(self.n)
        file_string = ""

        if self.seq_idxs is None:
            chosen_feed_idxs = torch.randperm(len(feeds))[0:len(IDs)]
        else:
            chosen_feed_idxs = self.seq_idxs

        chosen_feeds = feeds[chosen_feed_idxs] if len(chosen_feed_idxs) > 1 else [feeds[chosen_feed_idxs]]
        print(f"Chosen Feeds: {chosen_feeds}")
        init_state = "1'b1"
        print("Note: Using default init_state of 1 for ALL LFSRs and NLFSRs")
        for i, ID in enumerate(IDs):
            # Initialize a new FSR module
            file_string += f"module {name}_{ID}(\n" \
                           f"\tinput                 clock, reset,\n" \
                           f"\toutput logic [{self.n-1}:0]    state\n);\n"

            #  Initialize xor feedback bit for FSR
            file_string += "\tlogic xor_feedback;\n"

            # Compute the assignment equation for feedback_bit
            feed_eq = f'state[0]'  # the LSB is always part of the feedback
            for j in range(1, self.n):
                if chosen_feeds[i][j] == '1':
                    feed_eq += f' ^ state[{j}]'
            file_string += f"\tassign xor_feedback = {feed_eq};\n\n"

            # If we are constructing an NLFSR, then we need extra logic for the OR gate that inserts the all 0 state
            if self.nonlinear:
                file_string += "\tlogic zero_detect;\n"
                # OR together all bits but the LSB
                zero_eq = f"state[{self.n-1}]"
                for j in range(self.n-2, 0, -1):
                    zero_eq += f" | state[{j}]"
                file_string += f"\tassign zero_detect = {zero_eq};\n\n"

            # Create variable for FSR's feedback bit
            file_string += "\tlogic feedback_bit;\n"
            if self.nonlinear:  # if NLFSR then we need the logic for dealing with the all-0 state insertion
                file_string += f"\tassign feedback_bit = zero_detect ? xor_feedback : ~state[0];\n\n"
            else:  # if LFSR, then the feedback is just the xor feedback bit
                file_string += "\tassign feedback_bit = xor_feedback;\n\n"

            # Initialize logic for the state register and end the module
            file_string += f"\talways_ff @(posedge clock) begin\n" \
                           f"\t\tif (reset == 1) state      <= #1 {init_state};\n" \
                           f"\t\telse begin\n" \
                           f"\t\t                state[{self.n-2}:0] <= #1 state[{self.n-1}:1];\n" \
                           f"\t\t                state[{self.n-1}]   <= #1 feedback_bit;\n" \
                           f"\t\tend\n" \
                           f"\tend\n" \
                           f"endmodule\n\n\n"

        return file_string
'''


# TODO: Update this class from PyTorch
'''
class Counter_RNS(RNS):
    # hardware
    def __init__(self, n, **kwargs):
        super().__init__(n)
        self.name = "Counter"
        self.legend = "Counter"
        self.is_hardware = True
        self._max_N = kwargs.get('max_N', int(2**n))

    def _gen_RN(self, N, shape, share, starts=None, runs=1):
        assert N <= self._max_N
        if starts is not None:
            raise NotImplementedError
        return torch.arange(N).repeat((runs, *shape, 1))

    def gen_verilog(self, IDs=None, verbose=True):
        assert self._max_N == int(2**self.n), "Counter verilog only works for counters that use their whole sequence."
        super().verilog_info(verbose)
        print("Warning: Counters are generated with a rev_state output (so that VDC can be used). This should be free.")
        file_string = f"module counter (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\toutput logic [{n - 1}:0] state,\n" \
                      f"\toutput logic [{n - 1}:0] rev_state\n);\n" \
                      f"\talways_comb begin\n" \
                      f"\t\tfor (int i = 0; i < {n}; i+=1) begin\n" \
                      f"\t\t\trev_state[i] = state[{n - 1}-i];\n" \
                      f"\t\tend\n" \
                      f"\tend\n\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif (reset == 1) state <= #1 'b0; else\n" \
                      f"\t\t                state <= #1 state +1;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"
        return file_string
'''

class VDC_RNS(RNS):
    """
    RNS class for a Van der Corput low discrepancy sequence source.
    """
    def __init__(self, n, **kwargs):
        super().__init__(n)

        pow2n = int(2**n)

        self.name = "VDC"
        self.legend = "VDC"
        self.is_hardware = True
        self._max_N = pow2n
        self.seq = kwargs.get('vdc_seq')
        if self.seq is None:
            self.seq = seq_utils.get_vdc(n, verbose=True)

    def _gen_RN(self, N, shape, share):
        assert N <= self._max_N
        assert share or sum(shape) == 1

        return np.tile(self.seq, reps=math.prod(shape)).reshape(*shape, N)

    def gen_verilog(self, IDs, verbose=True):
        super().verilog_info(verbose)
        file_string = f"module counter (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\toutput logic [{self.n-1}:0] state,\n" \
                      f"\toutput logic [{self.n-1}:0] rev_state\n);\n" \
                      f"\talways_comb begin\n" \
                      f"\t\tfor (int i = 0; i < {self.n}; i+=1) begin\n" \
                      f"\t\t\trev_state[i] = state[{self.n-1}-i];\n" \
                      f"\t\tend\n" \
                      f"\tend\n\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif (reset == 1) state <= 'b0; else\n" \
                      f"\t\t                state <= state + 1;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"
        return file_string

    def gen_RN_posneg(self):
        pos_neg = np.empty((2, len(self.seq)), dtype=int)
        pos_neg[0] = self.seq
        pos_neg[1] = int(2 ** self.n) - 1 - self.seq  # inverting all bits is the same as doing Rs= (2^n -1-Rs)
        return pos_neg


# TODO: Update this class from PyTorch
'''
class LFSR_RNS(FSR_RNS):
    def __init__(self, n, **kwargs):
        kwargs['nonlinear'] = False
        super().__init__(n, **kwargs)
'''


# TODO: Update this class from PyTorch
'''
class NLFSR_RNS(FSR_RNS):
    def __init__(self, n, **kwargs):
        kwargs['nonlinear'] = True
        super().__init__(n, **kwargs)
'''


if __name__ == '__main__':
    pass
