"""Implementation of matrix embedding.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import scipy.optimize


def efficiency(
    alpha: float
) -> float:
    """Approximates embedding efficiency for Hamming code.

    Assumes last block to be padded.

    :param alpha: embedding rate in bits per element
    :type alpha: float
    :return: embedding efficiency in bits per change
    :rtype: float

    :Example:

    >>> e = cl.coding.efficiency(0.4)  # e at alpha=0.4
    """
    assert alpha >= 1e-3, 'instable for very low embedding rates'

    def mse_p(p) -> float:
        """SS error from blocksize."""
        return (p / (2**p - 1) - alpha)**2

    p = scipy.optimize.fminbound(mse_p, 1, 100, xtol=1e-4)
    return p / (1 - 2.**-p)
