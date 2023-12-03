"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np


def get_p(
    lbda: float,
    *rhos: np.ndarray,
) -> np.ndarray:
    """
    Calculate probability of embedding into each DCT coefficient location.
    This probability follows the Boltzmann distribution, also called Gibbs distribution.

    p_i = Pr(Y_i = y_i) = 1 / Z * exp( -lambda D(X, y_i X_~i )

    where
        x_i is the original value at the i-th position,
        y_i = argmin_{z \in {x_i - 1, x_i + 1}} D(X, zX_~i) is the modified value at the i-th position,
        Z is a normalization factor,
        y_i X_~i denotes the cover image whose i-th value has been modified to Y_i = y_i and all other pixels remain unchanged, and
        lambda is a constant.

    The equation is from
    Patrick Bas et al. "Break Our Steganographic System: The Ins and Outs of Organizing BOSS", IH 2011.
    Originally described in the STC paper by
    Filler et al. "Minimizing Additive Distortion in Steganography Using Syndrome-Trellis Codes", TIFS 2011.

    :param rho1: distortion of specific embedding choice, e.g. embedding +1 or embedding -1
    :param rho2: distortion of opposite embedding choice, needed for normalization
    :param lbda: constant
    :return: probability of embedding into each location, same shape as rho1
    """
    # denominator (forced left-associativity)
    denum = 1
    for rho in rhos:
        denum += np.exp(-lbda * rho)
    #
    return np.exp(-lbda * rhos[0]) / denum
