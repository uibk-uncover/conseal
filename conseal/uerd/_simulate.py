"""Simulation of UERD steganographic embedding.

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np

from ._costmap import compute_distortion
from ..simulate import _ternary
from .. import tools


def simulate_single_channel(
    cover_dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
    embedding_rate: float,
    payload_mode: str,
    wet_cost: float = 10**13,
    seed: int = None,
) -> np.ndarray:
    """Simulate embedding into a single channel.

    :param cover_dct_coeffs: array of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: np.ndarray
    :param quantization_table: array of shape [8, 8]
    :type quantization_table: np.ndarray
    :param embedding_rate: embedding rate
    :type embedding_rate: float
    :param payload_mode: "bpp" or "bpnzAC"
    :type payload_mode: str
    :param wet_cost: constant what the cost for wet pixel is
    :type wet_cost: float
    :param seed: random seed for STC simulator
    :type seed: int
    :return: stego DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: np.ndarray
    """

    if np.isclose(embedding_rate, 0):
        return cover_dct_coeffs

    num_vertical_blocks, num_horizontal_blocks = cover_dct_coeffs.shape[:2]

    # Compute distortion
    rho_p1, rho_m1 = compute_distortion(
        cover_dct_coeffs, quantization_table, wet_cost)

    # Determine number of available coefficients
    if "bpp" == payload_mode:
        n = cover_dct_coeffs.size
    elif "bpnzAC" == payload_mode:
        n = tools.dct.nzAC(cover_dct_coeffs)
    else:
        raise ValueError("Unknown payload mode")

    # Catch no nzAC (fails otherwise)
    if n == 0:
        raise ValueError('There are no non-zero AC coefficients for embedding')

    # simulator
    (pChangeP1, pChangeM1), lbda = _ternary.probability(
        rhoP1=rho_p1,
        rhoM1=rho_m1,
        alpha=embedding_rate,
        n=n,
    )
    delta_dct_coeffs = _ternary.simulate(
        pChangeP1=pChangeP1,
        pChangeM1=pChangeM1,
        seed=seed,
    )

    return cover_dct_coeffs + delta_dct_coeffs
