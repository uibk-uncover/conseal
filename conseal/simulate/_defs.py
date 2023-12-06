"""

The equation is from

P. Bas, et al. "Break Our Steganographic System: The Ins and Outs of Organizing BOSS", IH 2011.

Originally described in the STC paper by
T. Filler, et al. "Minimizing Additive Distortion in Steganography Using Syndrome-Trellis Codes", TIFS 2011.


Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np


def get_p(
    lbda: float,
    *rhos: np.ndarray,
) -> np.ndarray:
    """Converts distortions into probabilities,
    using Boltzmann-Gibbs distribution

    For more details, see `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#embedding-simulation>`__.

    :param rhos: distortion of embedding choices, e.g. embedding +1 or embedding -1
    :type rhos: tuple
    :param lbda: parameter value
    :type lbda: float
    :param p_pm1: probability tensor for changes associated to rhos[0]
        of an arbitrary shape
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # denominator (forced left-associativity)
    denum = 1
    for rho in rhos:
        denum += np.exp(-lbda * rho)
    #
    return np.exp(-lbda * rhos[0]) / denum
