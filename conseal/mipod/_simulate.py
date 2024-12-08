"""Implementation of MiPOD steganographic embedding.

V. Sedighi, R. Cogranne and J. Fridrich
"Content-Adaptive Steganography by Minimizing Statistical Detectability"
IEEE Transactions on Information Forensics and Security
Vol. 11, no. 2, pp. 221-234, Feb. 2016.
http://dde.binghamton.edu/vsedighi/pdf/TIFS2015_Content_Adaptive_Steganography_by_Minimizing_Statistical_Detectability.pdf
Code inspired by Sideghi's Matlab implementation: https://dde.binghamton.edu/download/stego_algorithms/download/MiPOD_matlab.zip

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import enum
import numpy as np
from typing import Tuple

from ._costmap import compute_cost
# from ..jmipod.simulate import ternary_probs

from . import _defs
from .. import tools


class Implementation(enum.Enum):
    """MiPOD implementation to choose from."""

    MiPOD_ORIGINAL = enum.auto()
    """Original MiPOD implementation by Remi Cogranne."""
    MiPOD_FIX_WET = enum.auto()
    """MiPOD implementation with wet-cost fixes."""


def probability(
    x0: np.ndarray,
    alpha: float,
) -> Tuple[Tuple[np.ndarray], float]:
    """Computes change probabilities for MiPOD embedding.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
    :type alpha: float
    :return: tuple ((p_p1, p_m1), lmbda), where
        p_p1 is the probability of +1 change,
        p_m1 is the probability of -1 change, and
        lbda is the determined lambda.
    :rtype: tuple

    :Example:

    >>> (p_p1, p_m1), _ = cl.mipod.probability(
    ...     x0=x0,
    ...     alpha=.4)
    """
    fisher_information = compute_cost(x0=x0)

    # Absolute payload in nats
    payload = alpha * np.prod(fisher_information.shape) * np.log(2)

    # change rate per pixel/selection channel
    p_flat = _defs.ternary_probs(fisher_information, payload, max_num_iterations=60, excess_values_num_iter=20)

    p = p_flat.reshape(x0.shape, order="F")

    return (p, p), alpha * np.prod(fisher_information.shape)


def simulate_single_channel(
    x0: np.ndarray,
    alpha: float,
    *,
    implementation: Implementation = Implementation.MiPOD_FIX_WET,
    seed: int = None,
) -> np.ndarray:
    """Simulates MiPOD steganography.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per pixel
    :type alpha: float
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: stego pixels,
        of shape [height, width, channels]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> x1 = cl.mipod.simulate_single_channel(
    ...     x0=x0,
    ...     alpha=.4,
    ...     seed=12345)
    """
    # rho is the fisher information
    rho = compute_cost(x0=x0)

    # Absolute payload in nats
    payload = alpha * np.prod(rho.shape) * np.log(2)

    # Beta is the change rate per pixel.
    # Beta is the selection channel.
    beta = _defs.ternary_probs(rho, payload, max_num_iterations=60, excess_values_num_iter=20)

    # The change rate for embedding +1 is beta and the change rate for embedding -1 is also beta.
    # Hence, the change of changing a pixel is 2 * beta.
    beta = 2 * beta

    rng = np.random.RandomState(seed)
    rand_change = rng.random_sample(x0.size)

    # Select cover elements to be modified by +- 1
    modify_pm_1 = rand_change < beta

    # Randomly select between +1 and -1 changes
    r = rng.random_sample(x0.size)
    delta_flat = tools.matlab_round(r[modify_pm_1]) * 2 - 1

    x1_flat = x0.copy().flatten(order="F").astype(int)
    x1_flat[modify_pm_1] += delta_flat.astype(int)
    x1 = x1_flat.reshape(x0.shape, order="F")

    # Take care of boundary cases
    if implementation == Implementation.MiPOD_ORIGINAL:
        x1[x1 > 255] = 253
        x1[x1 < 0] = 2
    elif implementation == Implementation.MiPOD_FIX_WET:
        x1[x1 > 255] = 254
        x1[x1 < 0] = 1
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')

    return x1.astype(np.uint8)
