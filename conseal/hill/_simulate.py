"""
Implementation of HILL steganographic embedding as described in:

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
from scipy.signal import convolve2d

from . import _costmap
from ..simulate import _ternary
from .. import tools


def simulate_single_channel(
    cover_spatial: np.ndarray,
    embedding_rate: float,
    wet_cost: float = 10**10,
    seed: int = None,
) -> np.ndarray:
    """Simulate embedding into a single channel.

    :param cover_spatial: uncompressed (pixel) cover image
        of shape [height, width]
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param embedding_rate: embedding rate in bits per pixel
    :type embedding_rate: float
    :param wet_cost: constant what the cost for wet pixel is
    :type wet_cost: float
    :param seed: random seed for STC simulator
    :type seed: int
    :return: stego pixels of shape [height, width, channels]
    :rtype: np.ndarray
    """
    if embedding_rate == 0:
        return cover_spatial

    # Compute distortion
    rho_p1, rho_m1 = _costmap.compute_cost_adjusted(
        cover_spatial,
        wet_cost,
    )

    # compute cover size
    n = cover_spatial.size

    # simulator
    (p_p1, p_m1), _ = _ternary.probability(
        rho_p1=rho_p1,
        rho_m1=rho_m1,
        alpha=embedding_rate,
        n=n,
    )
    delta = _ternary.simulate(
        p_p1=p_p1,
        p_m1=p_m1,
        seed=seed,
    )

    return cover_spatial + delta.astype('uint8')
