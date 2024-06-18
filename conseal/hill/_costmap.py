#!/usr/bin/env python3
"""
Implementation of HILL steganographic embedding.

Code inspired by https://github.com/daniellerch/stegolab.

A New Cost Function for Spatial Image Steganography
B. Li, M. Wang, J. Huang, X. Li.
"""

import numpy as np
import scipy.signal
import typing

from .. import tools


def compute_cost(
    cover_spatial: np.ndarray,
) -> np.ndarray:
    """Computes HILL cost.

    :param cover_spatial: uncompressed (pixel) cover image of shape [height, width]
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: cost of the same shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # high-pass filter
    H_KB = np.array([
        [-1,  2, -1],
        [ 2, -4,  2],
        [-1,  2, -1]
    ], dtype='float32')
    I1 = scipy.signal.convolve2d(
        cover_spatial, H_KB,
        mode='same', boundary='symm',
    )

    # low-pass filter 1
    L1 = np.ones((3, 3), dtype='float32') / 3**2
    I2 = scipy.signal.convolve2d(
        np.abs(I1), L1,
        mode='same', boundary='symm',
    )

    # low-pass filter 2
    L2 = np.ones((15, 15), dtype='float32')/15**2
    I2[I2 < tools.EPS] = tools.EPS
    I3 = scipy.signal.convolve2d(
        1./(I2), L2,
        mode='same', boundary='symm',
    )

    #
    return I3


def compute_cost_adjusted(
    cover_spatial: np.ndarray,
    wet_cost: float = 1e10,
) -> typing.Tuple[np.ndarray]:
    """Computes HILL cost with wet-cost adjustments.

    :param cover_spatial: uncompressed (pixel) cover image of shape [height, width]
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: 2-tuple (rho_p1, rho_m1), each of which is of the same shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # process input
    cover_spatial = cover_spatial.astype('float32')

    # Compute costmap
    rho = compute_cost(
        cover_spatial=cover_spatial
    )

    # Assign wet cost
    rho[np.isinf(rho) | np.isnan(rho) | (rho > wet_cost)] = wet_cost

    # Do not embed +1 if the pixel has max value
    rho_p1 = np.copy(rho)
    rho_p1[cover_spatial >= 255] = wet_cost

    # Do not embed -1 if the pixel has min value
    rho_m1 = np.copy(rho)
    rho_m1[cover_spatial <= 0] = wet_cost

    # return costs
    return rho_p1, rho_m1
