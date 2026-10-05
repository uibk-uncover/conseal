#!/usr/bin/env python3
"""
Implementation of S-UNIGARD steganographic embedding.

Selection-Channel-Aware Rich Model for Steganalysis of Digital Images.
T. Denemark, V. Sedighi, V. Holub, R. Cogranne, J. Fridrich.
http://www.ws.binghamton.edu/Fridrich/Research/maxSRM-final.pdf
"""


import numpy as np
from typing import List, Tuple

from .. import suniward


def gabor(
    num_points: int = 11,
    *,
    num_angles: int = 16,
    num_phases: int = 2,
    wavelength: float = 2.,
    gamma: float = .5,
    sigma: float = 1.,
) -> List[np.ndarray]:
    """Get 2D Gabor filter bank.

    :param num_points:
    :type num_points:
    :param num_angles:
    :type num_angles:
    :param num_phases:
    :type num_phases:
    :param wavelength:
    :type wavelength:
    :param gamma:
    :type gamma:
    :param sigma:
    :type sigma:
    :return:
    :rtype:

    :Example:

    >>> # TODO
    """
    # process parameters
    π = np.pi
    θs = -np.linspace(0, π, num_angles, endpoint=False).astype('float64')
    ϕs = np.linspace(0, π, num_phases, endpoint=False).astype('float32')
    cos_θs, sin_θs = np.cos(θs), np.sin(θs)
    λ = wavelength
    γ = gamma
    σ = sigma

    # produce grid
    xy = np.arange(num_points).astype('float32') - num_points//2
    x, y = np.meshgrid(xy, xy)

    # produce filters
    filters = []
    for ϕ in ϕs:
        for i, θ in enumerate(θs):
            # Eq. 2
            u = x * cos_θs[i] + y * sin_θs[i]
            # Eq. 3
            v = -x * sin_θs[i] + y * cos_θs[i]
            # Eq. 1
            Gxy = (
                np.exp(-(u**2 + γ**2 * v**2) / (2 * σ**2)) *
                np.cos(2*π*u/λ + ϕ)
            )
            filters.append(Gxy - np.mean(Gxy))
    return filters


def compute_cost(
    x0: np.ndarray,
    *,
    sigma: float = 1,
    filters: List[np.ndarray] = None,
    **kw
) -> np.ndarray:
    """Computes S-SUNIGARD cost.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param sigma: stabilizing constant. Small sigmas make the embedding very sensitive to the image content. Large sigmas smooth out the embedding change probabilities.
    :type sigma: float
    :param filters: filters to use, by default Gabor filters
    :type filters:
    :return: cost for +-1 change,
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """  # noqa: E501
    # get Gabor filters
    if filters is None:
        filters = gabor(**kw)
    # compute cost
    return suniward._costmap.compute_cost(x0=x0, sigma=sigma, filters=filters)


def compute_cost_adjusted(
    x0: np.ndarray,
    *,
    wet_cost: float = 10**8,
    **kw,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes S-UNIWARD cost with wet-cost adjustments.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: costs for +1 and -1 changes,
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.suniward.compute_cost_adjusted(x0=x0)
    """
    assert len(x0.shape) == 2, 'single channel expected'

    # process input
    x0 = x0.astype('float64')

    # Calculate costmap
    rho = compute_cost(x0=x0, **kw)

    # Assign wet cost
    rho[np.isinf(rho) | np.isnan(rho) | (rho > wet_cost)] = wet_cost
    # rho[(rho > wet_cost) | np.isnan(rho)] = wet_cost

    # Do not embed +1 if the pixel has max value
    rho_p1 = np.copy(rho)
    rho_p1[x0 == 255] = wet_cost

    # Do not embed -1 if the pixel has min value
    rho_m1 = np.copy(rho)
    rho_m1[x0 == 0] = wet_cost

    # return costs
    return rho_p1, rho_m1

