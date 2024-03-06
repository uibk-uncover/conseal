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
    c_res,
    s_res,
    sigma: float,
    gamma: float,
) -> float:
    c_w = (sigma + np.sqrt(np.sum(c_res**2)))**-gamma
    s_w = (sigma + np.sqrt(np.sum(s_res**2)))**-gamma
    Vc = (c_w + s_w)
    return Vc


@jit(nopython=True, fastmath=True, nogil=False, cache=True, parallel=False)
def GetLocalDistortion(
    C_resVect,
    S_resVect,
    sigma: float,
    gamma: float,
) -> float:
    D = 0.
    assert len(C_resVect) == 6, 'invalid C_resVect size'
    assert len(S_resVect) == 6, 'invalid S_resVect size'
    D += GetLocalPotential(C_resVect[0:3], S_resVect[0:3], sigma, gamma)
    D += GetLocalPotential(C_resVect[1:4], S_resVect[1:4], sigma, gamma)
    D += GetLocalPotential(C_resVect[2:5], S_resVect[2:5], sigma, gamma)
    D += GetLocalPotential(C_resVect[3:6], S_resVect[3:6], sigma, gamma)
    return D


@jit(nopython=True, fastmath=True, nogil=False, cache=True, parallel=False)
def compute_cost(
    cover: np.ndarray,
    coverPadded: np.ndarray,
    sigma: float = 1,
    gamma: float = 1,
) -> typing.Tuple[np.ndarray]:
    """Computes raw cost.

    Unlike most other distortions, HUGO is truly directional.


    """

    # embedding costs initialization
    responseP1 = np.array([0, 0, -1, +1, 0, 0])

    # declare costs
    rhoM1 = np.zeros(cover.shape, dtype='double')
    rhoP1 = np.zeros(cover.shape, dtype='double')

    # create residual
    C_Rez_H = coverPadded[:, :-1] - coverPadded[:, 1:]
    C_Rez_V = coverPadded[:-1, :] - coverPadded[1:, :]
    C_Rez_Diag = coverPadded[:-1, :-1] - coverPadded[1:, 1:]
    C_Rez_MDiag = coverPadded[:-1, 1:] - coverPadded[1:, :-1]

    # iterate over elements in the sublattice
    for row in prange(cover.shape[0]):
        for col in prange(cover.shape[1]):

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
            rhoM1[row, col] = D_M1
            rhoP1[row, col] = D_P1

    return rhoP1, rhoM1


def compute_cost_adjusted(
    cover: np.ndarray,
    *,
    sigma: float = 1,
    gamma: float = 1,
    wet_cost: float = 1e8,
):
    # cover to double
    cover = cover.astype('double')

    # create mirror padded cover image
    coverPadded = np.pad(cover, [[3]*2, [3]*2], 'symmetric')

    # call
    rho_p1, rho_m1 = compute_cost(cover, coverPadded, sigma, gamma)

    # adjust embedding costs
    rho_m1[rho_m1 > wet_cost] = wet_cost  # truncate
    rho_p1[rho_p1 > wet_cost] = wet_cost
    rho_p1[cover == 255] = wet_cost  # do not embed +1, if the pixel has max value
    rho_m1[cover == 0] = wet_cost  # do not embed -1, if the pixel has min value

    return rho_p1, rho_m1
