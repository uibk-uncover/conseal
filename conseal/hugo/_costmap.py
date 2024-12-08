#!/usr/bin/env python3
"""
Implementation of HUGO steganographic embedding.

Code inspired by Holub's Matlab implementation: https://dde.binghamton.edu/download/stego_algorithms/

Using High-Dimensional Image Models to Perform Highly Undetectable Steganography.
T. Pevny, T. Filler and P. Bas. https://hal.science/hal-00541353/document
"""


from numba import jit, prange
import numpy as np
import typing


@jit(nopython=True, fastmath=True, nogil=False, cache=True, parallel=False)
def GetLocalPotential(
    c_res: np.ndarray,
    s_res: np.ndarray,
    sigma: float,
    gamma: float,
) -> float:
    """"""
    c_w = (sigma + np.sqrt(np.sum(c_res**2)))**-gamma
    s_w = (sigma + np.sqrt(np.sum(s_res**2)))**-gamma
    Vc = (c_w + s_w)
    return Vc


@jit(nopython=True, fastmath=True, nogil=False, cache=True, parallel=False)
def GetLocalDistortion(
    c_res: np.ndarray,
    s_res: np.ndarray,
    sigma: float,
    gamma: float,
) -> float:
    """"""
    D = 0.
    assert len(c_res) == 6, 'invalid c_res size'
    assert len(s_res) == 6, 'invalid S_resVect size'
    D += GetLocalPotential(c_res[0:3], s_res[0:3], sigma, gamma)
    D += GetLocalPotential(c_res[1:4], s_res[1:4], sigma, gamma)
    D += GetLocalPotential(c_res[2:5], s_res[2:5], sigma, gamma)
    D += GetLocalPotential(c_res[3:6], s_res[3:6], sigma, gamma)
    return D


@jit(nopython=True, fastmath=True, nogil=False, cache=True, parallel=False)
def compute_cost(
    x0: np.ndarray,
    x0_padded: np.ndarray,
    sigma: float = 1,
    gamma: float = 1,
) -> typing.Tuple[np.ndarray]:
    """Computes HUGO cost.

    Unlike most other distortions, HUGO is truly directional.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param x0_padded:
    :type x0_padded: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param sigma: parameter
    :type sigma: float
    :param gamma: parameter
    :type gamma: float
    :return: cost of the same shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # embedding costs initialization
    response_p1 = np.array([0, 0, -1, +1, 0, 0])

    # declare costs
    rho_p1 = np.zeros(x0.shape, dtype='double')
    rho_m1 = np.zeros(x0.shape, dtype='double')

    # create residual
    d0_h = x0_padded[:, :-1] - x0_padded[:, 1:]
    d0_v = x0_padded[:-1, :] - x0_padded[1:, :]
    d0_d = x0_padded[:-1, :-1] - x0_padded[1:, 1:]
    d0_m = x0_padded[:-1, 1:] - x0_padded[1:, :-1]

    # iterate over elements in the sublattice
    for row in prange(x0.shape[0]):
        for col in prange(x0.shape[1]):

            d_p1, d_m1 = 0., 0.

            # horizontal
            x0_sub = d0_h[row+3, col:col+6]
            d_m1 += GetLocalDistortion(x0_sub, x0_sub-response_p1, sigma, gamma)
            d_p1 += GetLocalDistortion(x0_sub, x0_sub+response_p1, sigma, gamma)

            # vertical
            x0_sub = d0_v[row:row+6, col+3]
            d_m1 += GetLocalDistortion(x0_sub, x0_sub-response_p1, sigma, gamma)
            d_p1 += GetLocalDistortion(x0_sub, x0_sub+response_p1, sigma, gamma)

            # diagonal
            x0_sub = np.array([d0_d[row+i, col+i] for i in range(6)])
            d_m1 += GetLocalDistortion(x0_sub, x0_sub-response_p1, sigma, gamma)
            d_p1 += GetLocalDistortion(x0_sub, x0_sub+response_p1, sigma, gamma)

            # minor diagonal
            x0_sub = np.array([d0_m[row+i, col+5-i] for i in range(6)])
            d_m1 += GetLocalDistortion(x0_sub, x0_sub-response_p1, sigma, gamma)
            d_p1 += GetLocalDistortion(x0_sub, x0_sub+response_p1, sigma, gamma)

            #
            rho_p1[row, col] = d_p1
            rho_m1[row, col] = d_m1

    #
    return rho_p1, rho_m1


def compute_cost_adjusted(
    x0: np.ndarray,
    *,
    sigma: float = 1,
    gamma: float = 1,
    wet_cost: float = 1e8,
):
    """Computes HUGO cost with wet-cost adjustments.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param sigma: additive parameter
    :type sigma: float
    :param gamma: exponent parameter
    :type gamma: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: costs for +1 and -1 changes
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    assert len(x0.shape) == 2, 'single channel expected'

    # cover to double
    x0 = x0.astype('double')

    # create mirror padded cover image
    x0_padded = np.pad(x0, [[3]*2, [3]*2], 'symmetric')

    # Compute costmap
    rho_p1, rho_m1 = compute_cost(
        x0=x0,
        x0_padded=x0_padded,
        sigma=sigma,
        gamma=gamma,
    )

    # Assign wet cost
    rho_p1[np.isinf(rho_p1) | np.isnan(rho_p1) | (rho_p1 > wet_cost)] = wet_cost
    rho_m1[np.isinf(rho_m1) | np.isnan(rho_m1) | (rho_m1 > wet_cost)] = wet_cost

    # Do not embed +-1 if the pixel has min/max value
    rho_p1[x0 >= 255] = wet_cost
    rho_m1[x0 <= 0] = wet_cost

    return rho_p1, rho_m1
