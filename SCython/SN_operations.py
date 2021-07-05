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

def get_SCC(Xs, Ys, pxs, pys):
    """
    :param Xs: List of SNs to measure SCC.
    :param Ys: List of other SN to measure SCC (must be same size as Xs)
    :param pxs: Xs' values if known  (value will be estimated if not given)
    :param pys: Ys' values if known  (value will be estimated if not given)
    :return:
    """
    # Do some checking
    assert Xs.shape == Ys.shape
    if pxs is not None and pys is not None:
        assert pxs.shape == pys.shape
        assert pxs.shape == Xs.shape[:-1], f"pxs shape: {pxs.shape} and Xs shape: {Xs.shape[:-1]} must match"

    # Estimate SN probabilities if they were not given
    if pxs is None:
        pxs = np.mean(Xs, dim=-1)
    if pys is None:
        pys = np.mean(Ys, dim=-1)

    # SN_length is presumed to be last dimension
    SN_length = Xs.shape[-1]

    # first get the covariance, the numerator of SCC
    covariances = np.sum(Xs * Ys, axis=-1)/SN_length - pxs*pys

    # SCC treats positive and negative covariances differently
    pos_covs = covariances > 0
    neg_covs = covariances < 0
    zero_covs = covariances == 0

    # Compute the SCC by dividing the covariance by the proper SCC normalizing factor
    covariances[pos_covs] /= np.minimum(pxs[pos_covs], pys[pos_covs]) - pxs[pos_covs]*pys[pos_covs]
    covariances[neg_covs] /= pxs[neg_covs]*pys[neg_covs] - np.clip(pxs[neg_covs] + pys[neg_covs] - 1, a_min=0, a_max=None)
    covariances[zero_covs] = 0

    return covariances
