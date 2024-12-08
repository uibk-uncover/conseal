"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np

from ._costmap import compute_cost_adjusted, Implementation
from .. import simulate
from .. import tools


def simulate_single_channel(
    x0: np.ndarray,
    y0: np.ndarray,
    qt: np.ndarray,
    alpha: float,
    *,
    wet_cost: float = 10**13,
    dtype: np.dtype = np.float64,
    implementation: Implementation = Implementation.JUNIWARD_ORIGINAL,
    generator: str = None,
    seed: int = None,
) -> np.ndarray:
    """Simulates J-UNIWARD embedding at an embedding rate into single-channel cover,
    and returns stego.

    J-UNIWARD was introduced in
    V. Holub, et al. Universal distortion function for steganography in an arbitrary domain.
    EURASIP JIS, 2014.

    The details of the methods are described in the
    `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#jpeg-universal-wavelet-relative-distortion-j-uniward>`__.

    :param x0: decompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param y0: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table
        of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
        in bits per nzAC coefficient
    :type alpha: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :param dtype: data type to use for distortion computation,
        float64 by default
    :type dtype: `np.dtype <https://numpy.org/doc/stable/reference/generated/numpy.dtype.html>`__
    :param implementation: choose J-UNIWARD implementation
    :type implementation: :class:`Implementation`
    :param generator: type of PRNG used by embedding simulator
    :type generator: `numpy.random.Generator <https://numpy.org/doc/stable/reference/random/generator.html>`__
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: quantized stego DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> jpeg1.Y = cl.juniward.simulate_single_channel(
    ...   y0=jpeg0.Y,  # DCT
    ...   qt=jpeg0.qt[0],  # QT
    ...   x0=im0.spatial[..., 0],  # decompressed
    ...   alpha=0.4,  # embedding rate
    ...   seed=12345)  # seed
    """
    # Count number of embeddable DCT coefficients
    nzAC = tools.dct.nzAC(y0)

    if nzAC == 0:
        raise ValueError('Expected non-zero AC coefficients')

    # Compute cost for embedding into the quantized DCT coefficients
    # of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    rho_p1, rho_m1 = compute_cost_adjusted(
        x0=x0,
        y0=y0,
        qt=qt,
        dtype=dtype,
        implementation=implementation,
        wet_cost=wet_cost,
    )

    # Rearrange from 4D to 2D
    rho_p1_2d = tools.dct.jpeglib_to_jpegio(rho_p1)
    rho_m1_2d = tools.dct.jpeglib_to_jpegio(rho_m1)

    # STC simulation
    delta_2d = simulate.ternary(
        rhos=(rho_p1_2d, rho_m1_2d),
        alpha=alpha,
        n=nzAC,
        generator=generator,
        seed=seed,
    )

    # Convert from 2D to 4D
    delta = tools.dct.jpegio_to_jpeglib(delta_2d)

    # stego = cover + stego noise
    return y0 + delta
