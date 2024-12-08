"""
Implementation of the UERD steganography method as described in

L. Guo, J. Ni, W. Su, C. Tang and Y.-Q. Shi
"Using Statistical Image Model for JPEG Steganography: Uniform Embedding Revisited"
IEEE Transactions on Information Forensics and Security, 2015
https://ieeexplore.ieee.org/document/7225122

The idea of uniform embedding distortion (UED) is to spread the changes across all the DCT coefficients (including the DC, zero and non-zero AC coefficients).

The uniform embedding revisited distortion (UERD) incorporates the complexity of the specific DCT block and the specific DCT mode.

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""  # noqa: E501

import numpy as np
from scipy.signal import convolve2d
import typing

from .. import tools


def compute_block_energies(
    y: np.ndarray,
    qt: np.ndarray,
) -> np.ndarray:
    """Compute block energy as described in Eq. 3

    :param y: quantized DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table,
        of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: block energies,
        of shape [num_vertical_blocks, num_horizontal_blocks]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rho = cl.uerd._costmap.compute_block_energies(
    ...   y0=jpeg0.Y,  # DCT
    ...   qt=jpeg0.qt[0])  # QT
    """

    num_vertical_blocks, num_horizontal_blocks = y.shape[:2]

    # Dequantize DCT coefficients
    y_deq = y * qt[None, None, :, :]

    # Flatten last dimension
    y_deq = y_deq.reshape(num_vertical_blocks, num_horizontal_blocks, -1)

    # Calculate block energies, excluding the DC coefficient
    block_energies = np.sum(np.abs(y_deq)[:, :, 1:], axis=2)

    return block_energies


def compute_cost(
    y0: np.ndarray,
    qt: np.ndarray,
) -> np.ndarray:
    """Compute the UERD cost.

    Check Eq. 4 of the paper.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table,
        of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: embedding cost,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # Compute block energies
    block_energies = compute_block_energies(y0, qt)
    num_vertical_blocks, num_horizontal_blocks = block_energies.shape

    # Compute energy in 8-neighborhood using a convolution
    neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    block_energies_in_neighborhood = convolve2d(block_energies, neighborhood, mode="same", boundary="symm")

    # Block rho of shape [num_vertical_blocks, num_horizontal_blocks]
    block_rho = block_energies + 0.25 * block_energies_in_neighborhood

    # Mode rho of shape [64]
    mode_rho = np.zeros(64, dtype=float)
    # DC
    mode_rho[0] = 0.5 * (qt[0, 1] + qt[1, 0])
    # AC
    mode_rho[1:] = qt.flatten()[1:]

    # Combine block and mode rho
    # Division by zero should result in inf cost
    rho = np.divide(
        mode_rho[None, None, :],
        block_rho[:, :, None],
        out=np.full(shape=(num_vertical_blocks, num_horizontal_blocks, 64), fill_value=np.inf, dtype=float),
        where=block_rho[:, :, None] > 0
    )

    rho = rho.reshape(num_vertical_blocks, num_horizontal_blocks, 8, 8)

    return rho


def compute_cost_adjusted(
    y0: np.ndarray,
    qt: np.ndarray,
    *,
    wet_cost: float = 10**13,
    avoid_saturated: bool = False,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute the adjusted J-UNIWARD cost for ternary embedding.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table,
        of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: cost for unembeddable coefficients
    :type wet_cost: float
    :param avoid_saturated: hard-sets blocks with saturated pixels to wet
    :type avoid_saturated: bool
    :return: embedding costs of +1 and -1 changes,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rho_p1, rho_m1 = cl.uerd.compute_cost_adjusted(
    ...   y0=jpeg0.Y,      # DCT
    ...   qt=jpeg0.qt[0])  # QT
    """
    # Compute embedding cost rho
    rho = compute_cost(y0, qt)

    # Adjust embedding costs
    rho[np.isinf(rho) | np.isnan(rho) | (rho > wet_cost)] = wet_cost
    if avoid_saturated:
        x0 = tools.decompress_channel(y0)  # decompress DCT
        rho[(tools.jpegio_to_jpeglib(x0) == 0).any(axis=(2, 3))] = wet_cost
        rho[(tools.jpegio_to_jpeglib(x0) == 255).any(axis=(2, 3))] = wet_cost

    # Do not embed +1 if the DCT coefficient has maximum value
    rho_p1 = np.copy(rho)
    rho_p1[y0 >= 1023] = wet_cost

    # Do not embed -1 if the DCT coefficient has minimum value
    rho_m1 = np.copy(rho)
    rho_m1[y0 <= -1023] = wet_cost

    return rho_p1, rho_m1
