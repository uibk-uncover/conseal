#!/usr/bin/env python3
"""
Implementation of WOW steganographic embedding.

Code inspired by Holub's Matlab implementation: https://dde.binghamton.edu/download/stego_algorithms/

Designing Steganographic Distortion Using Directional Filters.
V. Holub, and J. Fridrich. http://www.ws.binghamton.edu/fridrich/research/WOW_rewritten_ver_WIFS_02.pdf
"""

import numpy as np
import scipy.signal
from typing import Tuple

from .. import tools


def compute_cost(
    x0: np.ndarray,
    *,
    p: float = -1,
) -> Tuple[np.ndarray]:
    """Computes WOW cost.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param p: parameter for reciprocal Hoelder norm
    :type p: float
    :return: cost for +-1 change,
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    assert len(x0.shape) == 2, 'single channel expected'

    # 2D wavelet filters (Daubechies 8)
    _, F = tools.spatial.daubechies8()

    # add padding
    padSize = np.max([f.shape for f in F])
    x0_padded = np.pad(x0, padSize, 'symmetric')

    # compute directional residual and suitability \xi for each filter
    xi = []
    for fIndex in range(3):

        # compute residual
        R = scipy.signal.convolve2d(
            x0_padded,
            F[fIndex],
            mode='same', boundary='symm'
        )

        # compute sustability
        xi.append(
            scipy.signal.convolve2d(
                np.abs(R),
                np.rot90(np.abs(F[fIndex]), k=2),
                mode='same', boundary='symm'
            )
        )
        # correct the suitability shift if filter size is even
        if F[fIndex].shape[0] % 2 != 0:
            xi[fIndex] = np.roll(xi[fIndex], 1, axis=0)
        if F[fIndex].shape[1] % 2 != 0:
            xi[fIndex] = np.roll(xi[fIndex], 1, axis=1)
        # remove padding
        x0_center = [
            (xi[fIndex].shape[i]-x0.shape[i]) >> 1
            for i in range(2)
        ]
        xi[fIndex] = xi[fIndex][
            x0_center[0]+1:-x0_center[0]+1,
            x0_center[1]+1:-x0_center[1]+1,
        ]

    # compute embedding costs \rho
    rho = np.sum([xi[i]**p for i in range(3)], axis=0)**(-1/p)
    # rho = np.expand_dims(rho, 2)
    return rho


def compute_cost_adjusted(
    x0: np.ndarray,
    wet_cost: float = 1e10,
    **kw,
) -> Tuple[np.ndarray]:
    """Computes WOW cost with wet-cost adjustments.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: costs for +1 and -1 changes,
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.wow.compute_cost_adjusted(x0=x0)
    """
    assert len(x0.shape) == 2, 'single channel expected'

    # process input
    x0 = x0.astype('float32')

    # Compute costmap
    rho = compute_cost(x0=x0, **kw)

    # Assign wet cost
    rho[np.isinf(rho) | np.isnan(rho) | (rho > wet_cost)] = wet_cost

    # Do not embed +1 if the pixel has max value
    rho_p1 = np.copy(rho)
    rho_p1[x0 >= 255] = wet_cost

    # Do not embed -1 if the pixel has min value
    rho_m1 = np.copy(rho)
    rho_m1[x0 <= 0] = wet_cost

    # return costs
    return rho_p1, rho_m1
