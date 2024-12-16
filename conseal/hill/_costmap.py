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
    x0: np.ndarray,
) -> np.ndarray:
    """Computes HILL cost.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: cost for +-1 change
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rho = cl.hill.compute_cost(x0=x0)
    """
    # high-pass filter
    H_KB = np.array([
        [-1, +2, -1],
        [+2, -4, +2],
        [-1, +2, -1]
    ], dtype='float32')
    I1 = scipy.signal.convolve2d(
        x0, H_KB,
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
    x0: np.ndarray,
    *,
    wet_cost: float = 1e10,
) -> typing.Tuple[np.ndarray]:
    """Computes HILL cost with wet-cost adjustments.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: costs for +1 and -1 changes
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.hill.compute_cost_adjusted(x0=x0)
    """
    assert len(x0.shape) == 2, 'single channel expected'
    # process input
    x0 = x0.astype('float32')

    # Compute costmap
    rho = compute_cost(x0=x0)

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
