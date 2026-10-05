"""
Implementation of the nsF5 steganography method as described in

J. Fridrich, T. Pevny, and J. Kodovsky.
"Statistically undetectable JPEG steganography: Dead ends, challenges, and opportunities"
Multimedia & Security, 2007
http://dde.binghamton.edu/kodovsky/pdf/Fri07-ACM.pdf

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""  # noqa: E501

import numpy as np
from typing import Tuple

from .. import tools


def probability(
    y0: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute the nsF5 probability map.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per nzAC coefficient
    :type alpha: float
    :return: tuple ((p_p1, p_m1), None), where
        p_p1 is the probability of +1 change
        p_m1 is the probability of -1 change.
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    assert len(y0.shape) == 4, 'DCT must have 4 dimensions'
    assert y0.shape[2] == y0.shape[3] == 8, 'expected blocks of size 8x8'

    # No embedding
    if np.isclose(alpha, 0):
        return np.zeros_like(y0)

    # Compute change rate on bound
    beta = tools.inv_entropy(alpha)

    # Number of nonzero AC DCT coefficients
    nzAC = tools.dct.nzAC(y0)
    if nzAC == 0:
        raise ValueError('There are no non-zero AC coefficients for embedding')

    # probability map
    p = np.ones(y0.shape, dtype='float64') * beta

    # do not change zeros or DC mode
    p[y0 == 0] = 0
    p[:, :, 0, 0] = 0

    # substract absolute value
    p_p1, p_m1 = p.copy(), p.copy()
    p_p1[y0 > 0] = 0
    p_m1[y0 < 0] = 0

    return (p_p1, p_m1), None


def compute_cost(
    y0: np.ndarray,
    *,
    wet_cost: float = 10**10
) -> np.ndarray:
    """Compute the nsF5 cost.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: probability of +-1 change,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    assert len(y0.shape) == 4, 'DCT must have 4 dimensions'
    assert y0.shape[2] == y0.shape[3] == 8, 'expected blocks of size 8x8'

    # cost
    rho = np.ones(y0.shape, dtype='float64')

    # do not change zeros or DC mode
    rho[y0 == 0] = wet_cost
    rho[:, :, 0, 0] = wet_cost

    return rho


def compute_cost_adjusted(
    y0: np.ndarray,
    *,
    wet_cost: float = 10**10,
) -> Tuple[np.ndarray]:
    """Compute nsF5 cost with wet-cost adjustments.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: probability of +-1 change,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.nsF5.compute_cost_adjusted(jpeg0.Y)
    >>> jpeg1.Y = jpeg0.Y + cl.simulate.ternary(rhos=rhos, alpha=.4, seed=12345)
    """
    # Compute costmap
    rho = compute_cost(y0=y0, wet_cost=wet_cost)

    # Assign wet cost
    rho[np.isinf(rho) | np.isnan(rho) | (rho > wet_cost)] = wet_cost

    # Subtract absolute value
    rho_p1, rho_m1 = rho.copy(), rho.copy()
    rho_p1[y0 > 0] = wet_cost
    rho_m1[y0 < 0] = wet_cost

    return rho_p1, rho_m1
