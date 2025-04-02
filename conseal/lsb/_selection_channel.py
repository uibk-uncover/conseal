"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from scipy.stats import rankdata


def probability(
    rhos: np.ndarray,
    alpha: float,
    e: float = 2
) -> np.ndarray:
    """Returns the probabilities for a selection channel steganography.

    A quick and dirty implementation. Will be updated in the future.
    See the code comments for more information.

    :param rhos: embedding costs
    :type rhos: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
    :type alpha: float
    :param e: embedding efficiency
    :type e: float
    :return: change probabilities
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    """
    # cost rank
    ranks = rankdata(rhos.reshape(-1), method='ordinal') - 1
    """smallest cost is 0, largest cost is N-1"""
    # probability of this cost being lower (out of 2)
    ps = [
        (rhos.size - rank) / (rhos.size - 1)
        for rank in ranks
    ]
    """normalize the ranks to [0,1]"""
    # probability of this cost being lower from k
    inv_alpha = np.floor(1 / alpha)  # block size
    ps = np.array([
        p**(inv_alpha - 1) / e
        for p in ps
    ]).reshape(rhos.shape)
    """extends from 2 to k.

    For e=2, this is

    k  p
    1  p^0/2=0.5 (behaves wierd for alpha>0.5)
    2  p^1/2
    3  p^2/2
    4  p^3/2

    Quick and dirty implementations. Will be updated in the future.

    Know limitations (for the moment):
    - assumes sampling with replacement (should be ok for large covers)
    """

    #
    return ps.reshape(rhos.shape)
