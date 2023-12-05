"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy
import numpy as np
import typing

from ._costmap import compute_distortion, Implementation
from .. import simulate
from .. import tools


def simulate_single_channel(
    cover_spatial: np.ndarray,
    cover_dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
    embedding_rate: float,
    wet_cost: float = 10**13,
    dtype: typing.Type = np.float64,
    implementation: Implementation = Implementation.JUNIWARD_ORIGINAL,
    generator: str = None,
    seed: int = None,
) -> np.ndarray:
    """J-UNIWARD embedding

    :param cover_spatial: spatial image of shape [height, width]
    :type cover_spatial: :class:`np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param cover_dct_coeffs: corresponding quantized DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: :class:`np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param quantization_table: quantization table of shape [8, 8]
    :type quantization_table: :class:`np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param embedding_rate: rate of bits to embed per nzAC coefficient
    :type embedding_rate: float
    :param wet_cost: cost for unembeddable coefficients
    :type wet_cost: float
    :param dtype: float32 or float64
    :type dtype: :class:`np.dtype <https://numpy.org/doc/stable/reference/generated/numpy.dtype.html>`_
    :param implementation: choose J-UNIWARD implementation
    :type implementation: :class:`Implementation`
    :param generator: type of random number generator passed on to the stego noise simulator
    :type generator: :class:`numpy.random.Generator`
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: stego DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: :class:`numpy.ndarray`
    """
    # Count number of embeddable DCT coefficients
    num_non_zero_AC_coeffs = tools.dct.nzAC(cover_dct_coeffs)

    if num_non_zero_AC_coeffs == 0:
        raise ValueError('Expected non-zero AC coefficients')

    # Compute cost for embedding into the quantized DCT coefficients, shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    rho_p1, rho_m1 = compute_distortion(
        cover_spatial=cover_spatial,
        cover_dct_coeffs=cover_dct_coeffs,
        quant_table=quantization_table,
        dtype=dtype,
        implementation=implementation,
        wet_cost=wet_cost,
    )

    # Rearrange from 4D to 2D
    rho_p1_2d = tools.dct.jpeglib_to_jpegio(rho_p1)
    rho_m1_2d = tools.dct.jpeglib_to_jpegio(rho_m1)

    # STC simulation
    stego_noise_dct_2d = simulate.ternary(
        rhoP1=rho_p1_2d,
        rhoM1=rho_m1_2d,
        alpha=embedding_rate,
        n=num_non_zero_AC_coeffs,
        generator=generator,
        seed=seed,
    )

    # Convert from 2D to 4D
    stego_noise_dct = tools.dct.jpegio_to_jpeglib(stego_noise_dct_2d)

    # stego = cover + stego noise
    stego_dct_coeffs = cover_dct_coeffs + stego_noise_dct

    return stego_dct_coeffs
