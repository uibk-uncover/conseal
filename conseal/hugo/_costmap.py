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
    cover_spatial: np.ndarray,
    cover_spatial_padded: np.ndarray,
    sigma: float = 1,
    gamma: float = 1,
) -> typing.Tuple[np.ndarray]:
    """Computes raw cost.

    Unlike most other distortions, HUGO is truly directional.

    :param cover_spatial: uncompressed (pixel) cover image
        of shape [height, width]
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param cover_spatial_padded:
    :type cover_spatial_padded: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param sigma:
    :type sigma: float
    :param gamma:
    :type gamma: float

    :Example:
    >>> # TODO
    """

    # embedding costs initialization
    responseP1 = np.array([0, 0, -1, +1, 0, 0])

    # declare costs
    rho_p1 = np.zeros(cover_spatial.shape, dtype='double')
    rho_m1 = np.zeros(cover_spatial.shape, dtype='double')

    # create residual
    C_Rez_H = cover_spatial_padded[:, :-1] - cover_spatial_padded[:, 1:]
    C_Rez_V = cover_spatial_padded[:-1, :] - cover_spatial_padded[1:, :]
    C_Rez_Diag = cover_spatial_padded[:-1, :-1] - cover_spatial_padded[1:, 1:]
    C_Rez_MDiag = cover_spatial_padded[:-1, 1:] - cover_spatial_padded[1:, :-1]

    # iterate over elements in the sublattice
    for row in prange(cover_spatial.shape[0]):
        for col in prange(cover_spatial.shape[1]):

            D_P1, D_M1 = 0., 0.

            # horizontal
            cover_sub = C_Rez_H[row+3, col:col+6]
            D_M1 += GetLocalDistortion(cover_sub, cover_sub-responseP1, sigma, gamma)
            D_P1 += GetLocalDistortion(cover_sub, cover_sub+responseP1, sigma, gamma)

            # vertical
            cover_sub = C_Rez_V[row:row+6, col+3]
            D_M1 += GetLocalDistortion(cover_sub, cover_sub-responseP1, sigma, gamma)
            D_P1 += GetLocalDistortion(cover_sub, cover_sub+responseP1, sigma, gamma)

            # diagonal
            cover_sub = np.array([C_Rez_Diag[row+i, col+i] for i in range(6)])
            D_M1 += GetLocalDistortion(cover_sub, cover_sub-responseP1, sigma, gamma)
            D_P1 += GetLocalDistortion(cover_sub, cover_sub+responseP1, sigma, gamma)

            # minor diagonal
            cover_sub = np.array([C_Rez_MDiag[row+i, col+5-i] for i in range(6)])
            D_M1 += GetLocalDistortion(cover_sub, cover_sub-responseP1, sigma, gamma)
            D_P1 += GetLocalDistortion(cover_sub, cover_sub+responseP1, sigma, gamma)

            #
            rho_p1[row, col] = D_P1
            rho_m1[row, col] = D_M1

    #
    return rho_p1, rho_m1


def compute_cost_adjusted(
    cover_spatial: np.ndarray,
    *,
    sigma: float = 1,
    gamma: float = 1,
    wet_cost: float = 1e8,
):
    """

    :param cover_spatial: uncompressed (pixel) cover image
        of shape [height, width]
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param sigma:
    :type sigma: float
    :param gamma:
    :type gamma: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float

    :Example:
    >>> # TODO
    """
    # cover to double
    cover_spatial = cover_spatial.astype('double')

    # create mirror padded cover image
    cover_spatial_padded = np.pad(cover_spatial, [[3]*2, [3]*2], 'symmetric')

    # Compute costmap
    rho_p1, rho_m1 = compute_cost(
        cover_spatial=cover_spatial,
        cover_spatial_padded=cover_spatial_padded,
        sigma=sigma,
        gamma=gamma,
    )

    # Assign wet cost
    rho_p1[np.isinf(rho_p1) | np.isnan(rho_p1) | (rho_p1 > wet_cost)] = wet_cost
    rho_m1[np.isinf(rho_m1) | np.isnan(rho_m1) | (rho_m1 > wet_cost)] = wet_cost

    # Do not embed +-1 if the pixel has min/max value
    rho_p1[cover_spatial >= 255] = wet_cost
    rho_m1[cover_spatial <= 0] = wet_cost

    return rho_p1, rho_m1
