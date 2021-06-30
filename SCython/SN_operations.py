import numpy as np

def get_SN_value(SN_array, bipolar):
    """
    Computes the value of each SN in an array. Assumes last dimension of SN_array corresponds to the SN bits.
    For instance, if SN_array is shape (5,2,32), then that means we have a 5x2 array of SNs of length 32.
    :param SN_array
    :param bipolar: whether or not bipolar format is used
    :return: SN values
    """
    if bipolar:
        values = 2*np.mean(SN_array, axis=-1) - 1
    else:
        values = np.mean(SN_array, axis=-1)

    return values
