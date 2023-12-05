"""Implementation of the UERD steganography method as described in

L. Guo, J. Ni, W. Su, C. Tang and Y.-Q. Shi
"Using Statistical Image Model for JPEG Steganography: Uniform Embedding Revisited"
IEEE Transactions on Information Forensics and Security, 2015
https://ieeexplore.ieee.org/document/7225122

The idea of uniform embedding distortion (UED) is to spread the changes across all the DCT coefficients (including the DC, zero and non-zero AC coefficients).

The uniform embedding revisited distortion (UERD) incorporates the complexity of the specific DCT block and the specific DCT mode.

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
from scipy.signal import convolve2d
import typing


def compute_block_energies(
    dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
) -> np.ndarray:
    """Compute block energy as described in Eq. 3

    :param dct_coeffs: dct coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param quantization_table: quantization table of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: block energies of shape [num_vertical_blocks, num_horizontal_blocks]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """

    num_vertical_blocks, num_horizontal_blocks = dct_coeffs.shape[:2]

    # Dequantize DCT coefficients
    dequantized_dct_coeffs = dct_coeffs * quantization_table[None, None, :, :]

    # Flatten last dimension
    dequantized_dct_coeffs = dequantized_dct_coeffs.reshape(num_vertical_blocks, num_horizontal_blocks, -1)

    # Calculate block energies, excluding the DC coefficient
    block_energies = np.sum(np.abs(dequantized_dct_coeffs)[:, :, 1:], axis=2)

    return block_energies


def compute_cost(
    cover_dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
) -> np.ndarray:
    """Compute embedding cost as described in Eq. 4

    :param cover_dct_coeffs: ndarray of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param quantization_table: ndarray of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: embedding cost of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # Compute block energies
    block_energies = compute_block_energies(cover_dct_coeffs, quantization_table)
    num_vertical_blocks, num_horizontal_blocks = block_energies.shape

    # Compute energy in 8-neighborhood using a convolution
    neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    block_energies_in_neighborhood = convolve2d(block_energies, neighborhood, mode="same", boundary="symm")

    # Block rho of shape [num_vertical_blocks, num_horizontal_blocks]
    block_rho = block_energies + 0.25 * block_energies_in_neighborhood

    # Mode rho of shape [64]
    mode_rho = np.zeros(64, dtype=float)
    # DC
    mode_rho[0] = 0.5 * (quantization_table[0, 1] + quantization_table[1, 0])
    # AC
    mode_rho[1:] = quantization_table.flatten()[1:]

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


def compute_distortion(
    cover_dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
    wet_cost: float = 10**13,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Computes the distortion rho_p1 and rho_m1.

    :param cover_dct_coeffs: array of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param quantization_table: ndarray of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: constant what the cost for wet pixel is
    :type wet_cost: float
    :return: 2-tuple (rho_p1, rho_m1), each of which is of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    # Compute embedding cost rho
    rho = compute_cost(cover_dct_coeffs, quantization_table)

    # Adjust embedding costs
    rho[np.isinf(rho) | np.isnan(rho) | (rho > wet_cost)] = wet_cost

    # Do not embed +1 if the DCT coefficient has maximum value
    rho_p1 = np.copy(rho)
    rho_p1[cover_dct_coeffs >= 1023] = wet_cost

    # Do not embed -1 if the DCT coefficient has minimum value
    rho_m1 = np.copy(rho)
    rho_m1[cover_dct_coeffs <= -1023] = wet_cost

    return rho_p1, rho_m1
