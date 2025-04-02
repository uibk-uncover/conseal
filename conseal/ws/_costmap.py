"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from scipy.signal import convolve2d


def compute_cost(x0: np.ndarray) -> np.ndarray:
    """Computes a adversarial cost against WS steganalysis.

    It transforms the WS residual using the exponential function.

    Intended for selection-channel steganography for simple comparison
    against vanilla LSBR performance.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :returns: cost against WS
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    """
    # flip
    x0_bar = x0 ^ 1
    x0 = x0.astype('float32')

    # predict
    f_kb = np.array([
        [-1, +2, -1],
        [+2,  0, +2],
        [-1, +2, -1],
    ], dtype='float32') / 4.
    x0_hat = convolve2d(
        x0, f_kb,
        mode='same', boundary='symm'
    )

    # calculate local WS residuals
    betas_hat = (x0 - x0_bar) * (x0 - x0_hat)

    # convert to distortion
    rhos = np.exp(-np.abs(betas_hat))
    """This is likely not optimal, but I had little time to test something better."""
    return rhos


def compute_cost_adjusted(
    x0,
    *,
    wet_cost: float = 1e10,
) -> np.ndarray:
    """Computes a adversarial cost against WS steganalysis.

    It transforms the WS residual using the exponential function.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :returns: cost against WS
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    """
    # compute cost
    rho = compute_cost(x0)


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
