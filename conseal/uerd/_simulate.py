"""Simulation of the UERD steganography method.

We thank Patrick Bas for sharing his implementation of UERD with us.

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np

from ._costmap import compute_cost_adjusted
from ..simulate import _ternary
from .. import tools


def simulate_single_channel(
    y0: np.ndarray,
    qt: np.ndarray,
    alpha: float,
    *,
    payload_mode: str = "bpnzAC",
    wet_cost: float = 10**13,
    seed: int = None,
) -> np.ndarray:
    """Simulates UERD embedding at an embedding rate into single-channel cover,
    and returns stego.

    UERD was introduced in
    L. Guo, et al. Using Statistical Image Model for JPEG Steganography: Uniform Embedding Revisited.
    IEEE TIFS, 2015.

    The details of the methods are described in the
    `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#uniform-embedding-revisited-distortion-uerd>`__.

    :param y0: quantized DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantized table of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate, in unit specified by payload_mode
    :type alpha: float
    :param payload_mode: unit used by embedding rate, either
        "bpc" (bits per DCT coefficient), or
        "bpnzAC" (bits per non-zero DCT AC coefficient).
    :type payload_mode: str
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: stego DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y = cl.uerd.simulate_single_channel(
    ...   y0=im_dct.Y,  # DCT
    ...   qt=im_dct.qt[0],  # QT
    ...   alha=0.4,  # alpha
    ...   seed=12345)  # seed
    """
    if np.isclose(alpha, 0):
        return y0

    # Compute distortion
    rhos = compute_cost_adjusted(
        y0=y0,
        qt=qt,
        wet_cost=wet_cost,
    )

    # Determine number of available coefficients
    if "bpc" == payload_mode:
        n = y0.size
    elif "bpnzAC" == payload_mode:
        n = tools.dct.nzAC(y0)
    else:
        raise ValueError("Unknown payload mode")

    # Catch no nzAC (fails otherwise)
    if n == 0:
        raise ValueError('There are no non-zero AC coefficients for embedding')

    # simulator
    ps, lbda = _ternary.probability(
        rhos=rhos,
        alpha=alpha,
        n=n,
    )

    delta = _ternary.simulate(
        ps=ps,
        seed=seed,
    )

    return y0 + delta
