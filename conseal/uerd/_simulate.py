"""Simulation of the UERD steganography method.

We thank Patrick Bas for sharing his implementation of UERD with us.

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

    :param cover_dct_coeffs: quantized DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param quantization_table: quantized table of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param embedding_rate: embedding rate, in unit specified by payload_mode
    :type embedding_rate: float
    :param payload_mode: unit used by embedding rate, either
        "bpc" (bits per DCT coefficient), or
        "bpnzAC" (bits per non-zero DCT AC coefficient).
    :type payload_mode: str
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :param seed: random seed for STC simulator
    :type seed: int
    :return: stego DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y = cl.uerd.simulate_single_channel(
    ...   cover_dct_coeffs=im_dct.Y,  # DCT
    ...   quantization_table=im_dct.qt[0],  # QT
    ...   embedding_rate=0.4,  # alpha
    ...   seed=12345)  # seed
    """
    if np.isclose(embedding_rate, 0):
        return cover_dct_coeffs

    # Compute distortion
    rho_p1, rho_m1 = compute_distortion(
        cover_dct_coeffs=cover_dct_coeffs,
        quantization_table=quantization_table,
        payload_mode=payload_mode,
        wet_cost=wet_cost,
    )

    # Determine number of available coefficients
    if "bpc" == payload_mode:
        n = cover_dct_coeffs.size
    elif "bpnzAC" == payload_mode:
        n = tools.dct.nzAC(cover_dct_coeffs)
    else:
        raise ValueError("Unknown payload mode")

    # Catch no nzAC (fails otherwise)
    if n == 0:
        raise ValueError('There are no non-zero AC coefficients for embedding')

    # simulator
    (p_p1, p_m1), lbda = _ternary.probability(
        rho_p1=rho_p1,
        rho_m1=rho_m1,
        alpha=embedding_rate,
        n=n,
    )
    delta_dct_coeffs = _ternary.simulate(
        p_p1=p_p1,
        p_m1=p_m1,
        seed=seed,
    )

    return cover_dct_coeffs + delta_dct_coeffs
