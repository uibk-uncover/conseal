"""

Author:Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np

from ._costmap import compute_cost_adjusted, Implementation
from .. import simulate
from .. import tools


def simulate_single_channel(
    cover_dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
    embedding_rate: float,
    wet_cost: float = 10**13,
    implementation: Implementation = Implementation.EBS_ORIGINAL,
    generator: str = None,
    seed: int = None,
) -> np.ndarray:
    """Simulates EBS embedding at an embedding rate into single-channel cover,
    and returns stego.

    EBS was introduced in
    C. Wang, et al. An efficient JPEG steganographic scheme based on block entropy of DCT coefficients.
    IEEE ICASSP, 2012.

    The details of the methods are described in the
    `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#entropy-block-stego>`__.

    :param cover_dct_coeffs: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param quantization_table: quantization table
        of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param embedding_rate: embedding rate,
        in bits per nzAC coefficient
    :type embedding_rate: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :param implementation: choose EBS implementation
    :type implementation: :class:`Implementation`
    :param generator: type of PRNG used by embedding simulator
    :type generator: `numpy.random.Generator <https://numpy.org/doc/stable/reference/random/generator.html>`__
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: quantized stego DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y = cl.ebs.simulate_single_channel(
    ...   cover_dct_coeffs=im_dct.Y,  # DCT
    ...   quantization_table=im_dct.qt[0],  # QT
    ...   embedding_rate=0.4,  # alpha
    ...   seed=12345)  # seed
    """
    # Count number of embeddable DCT coefficients
    N, M, _, _ = cover_dct_coeffs.shape
    if implementation == Implementation.EBS_ORIGINAL:
        num_DCT_coeffs = cover_dct_coeffs.size - 4*N*M
    elif implementation == Implementation.EBS_FIX_WET:
        num_DCT_coeffs = cover_dct_coeffs.size
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')

    # Compute cost for embedding into the quantized DCT coefficients
    # of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    rho_p1, rho_m1 = compute_cost_adjusted(
        cover_dct_coeffs=cover_dct_coeffs,
        quantization_table=quantization_table,
        implementation=implementation,
        wet_cost=wet_cost,
    )

    # STC simulation
    stego_noise_dct = simulate.ternary(
        rho_p1=rho_p1,
        rho_m1=rho_m1,
        alpha=embedding_rate,
        n=num_DCT_coeffs,
        generator=generator,
        seed=seed,
    )

    # stego = cover + stego noise
    stego_dct_coeffs = cover_dct_coeffs + stego_noise_dct

    return stego_dct_coeffs
