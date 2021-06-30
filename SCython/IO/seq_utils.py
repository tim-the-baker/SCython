import numpy as np
import os

# Useful directories
SOBOL_SEQ_DIR = r'C:\Users\Tim\Documents\Michigan Research Data\Sobol Seq Data'
LFSR_SEQ_DIR = r'C:\Users\Tim\Documents\Michigan Research Data\LFSR Seq Data'
LFSR_FEED_DIR = r'C:\Users\Tim\Documents\Michigan Research Data\LFSR Feed Data'

def get_vdc(n, verbose=True):
    """
    Generates the VDC sequence according to a given precision
    :param n: the desired precision, in bits
    :param verbose: when True, a message is printed that indicates this method is run. This method is slow and should
    only be run once during the entire simulation rather than once per simulation run.
    :return: the n-bit VDC sequence
    """
    if verbose:
        print(f"Generating {n}-bit VDC sequence")
    pow2n = int(2**n)
    seq = np.zeros(pow2n, dtype=np.int)
    for idx in range(pow2n):
        bin_rep = np.binary_repr(idx, n)
        seq[idx] = sum([int(2**b_idx * int(bit)) for b_idx, bit in enumerate(bin_rep)])
    return seq


# TODO: Convert this file from PyTorch to NumPy
'''
def get_LFSR_seqs(n: int, feedback: str, extended: bool = False, verbose: bool = True) -> torch.tensor:
    """
    :param n: bit-width of LFSR
    :param feedback: LFSR feedback type. 'e': external feedback, 'i': internal feedback, 'a': all/both feedback types.
    :param extended: if True, use a LFSR that has the all 0 state inserted. LFSR becomes a NLFSR.
    :return: corresponding list of LFSR seqs.
    """
    if verbose:
        print(f"Loading {n}-bit {feedback} LFSR sequences")
    all_seqs = np.load(LFSR_SEQ_DIR + r'\n%d_e%d_p%d.npy' % (n, extended, 0))
    all_seqs = torch.tensor(all_seqs, dtype=torch.int)
    if feedback == 'a' or extended:
        return all_seqs
    elif feedback == 'e':
        return all_seqs[len(all_seqs)//2:]  # second half are the external LFSR seqs
    elif feedback == 'i':
        return all_seqs[0:len(all_seqs)]  # first half are the internal LFSR seqs
    else:
        raise ValueError(f"feedback should be 'e' (external) 'i' (internal) or 'a' (all). Given feedback: {feedback}")


def get_LFSR_feeds(n: int):
    feeds = np.load(LFSR_FEED_DIR + fr'\\n{n}.npy')
    return feeds
'''
