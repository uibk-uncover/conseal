"""Implementation of J-MiPOD steganographic embedding.

R. Cogranne, Q. Giboulot, P. Bas
"Steganography by Minimizing Statistical Detectability: The cases of JPEG and Color Images"
ACM IH&MMSec, 2020, Denver, CO, USA, pp. 161-167
https://doi.org/10.1145/3369412.3395075

Based on the prior work for spatial domain images:
V. Sedighi, R. Cogranne and J. Fridrich
"Content-Adaptive Steganography by Minimizing Statistical Detectability"
IEEE Transactions on Information Forensics and Security
Vol. 11, no. 2, pp. 221-234, Feb. 2016.

Code inspired by the reference Matlab implementation: https://codeocean.com/capsule/7800700/tree/v4

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Tuple

from ._costmap import compute_cost
from . import _defs
from .. import tools


def probability(
    y0: np.ndarray,
    qt: np.ndarray,
    alpha: float,
) -> Tuple[Tuple[np.ndarray], float]:
    """Computes change probabilities for J-MiPOD embedding.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table,
        of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per non-zero AC DCT coefficient
    :type alpha: float
    :return: tuple ((p_p1, p_m1), payload), where
        p_p1 is the probability of +1 change,
        p_m1 is the probability of -1 change, and
        payload is the absolute payload in nats.
    :rtype: tuple

    :Example:

    >>> (p_p1, p_m1), _ = cl.jmipod.probability(
    ...     y0=y0,
    ...     qt=qt,
    ...     alpha=.4)
    """
    fisher_information = compute_cost(y0=y0, qt=qt)

    # Number of embeddable (non-zero AC) DCT coefficients
    nzAC = tools.dct.nzAC(y0)

    # Absolute payload in nats
    payload = alpha * nzAC * np.log(2)

    # change rate per DCT coefficient/selection channel
    p_flat = _defs.ternary_probs(fisher_information, payload, max_num_iterations=30, excess_values_num_iter=10)

    p = p_flat.reshape(fisher_information.shape, order="F")

    return (p, p), payload


def simulate_single_channel(
    y0: np.ndarray,
    qt: np.ndarray,
    alpha: float,
    *,
    seed: int = None,
) -> np.ndarray:
    """Simulates J-MiPOD steganography.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table,
        of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per non-zero AC DCT coefficient
    :type alpha: float
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: quantized stego DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> jpeg1.Y = cl.jmipod.simulate_single_channel(
    ...     y0=jpeg0.Y,
    ...     qt=jpeg0.qt[0],
    ...     alpha=.4,
    ...     seed=12345)
    """
    # rho is the fisher information
    rho = compute_cost(y0=y0, qt=qt)

    # Number of embeddable (non-zero AC) DCT coefficients
    nzAC = tools.dct.nzAC(y0)

    # Absolute payload in nats
    payload = alpha * nzAC * np.log(2)

    # Beta is the change rate per DCT coefficient.
    # Beta is the selection channel.
    beta = _defs.ternary_probs(rho, payload, max_num_iterations=30, excess_values_num_iter=10)

    # The change rate for embedding +1 is beta and the change rate for embedding -1 is also beta.
    # Hence, the change of changing a coefficient is 2 * beta.
    beta = 2 * beta

    rng = np.random.RandomState(seed)
    rand_change = rng.random_sample(y0.size)

    # Select cover elements to be modified by +- 1
    modify_pm_1 = rand_change < beta

    # Randomly select between +1 and -1 changes
    r = rng.random_sample(y0.size)
    delta_flat = tools.matlab_round(r[modify_pm_1]) * 2 - 1

    y0_2d = tools.dct.jpeglib_to_jpegio(y0)
    y1_flat = y0_2d.copy().flatten(order="F").astype(int)
    y1_flat[modify_pm_1] += delta_flat.astype(int)
    y1_2d = y1_flat.reshape(y0_2d.shape, order="F")

    # Take care of boundary cases
    y1_2d = np.clip(y1_2d, a_min=-1024, a_max=1023)

    y1 = tools.dct.jpegio_to_jpeglib(y1_2d)

    return y1.astype(y0.dtype)
