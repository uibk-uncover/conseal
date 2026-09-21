"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np

from . import _defs
from ..mipod._costmap import estimate_variance
from .. import tools

__all__ = [
    'estimate_variance',
    'compute_cost',
]


def compute_cost(
    y0: np.ndarray,
    qt: np.ndarray,
) -> np.ndarray:
    """Produces J-MiPOD cost. Not a cost like HILL, but a Fisher information.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qt: quantization table,
        of shape [8, 8]
    :type qt: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: cost for +-1 change,
        of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> fi = cl.jmipod.compute_cost(y0=y0, qt=qt)
    """
    y0 = y0.astype('float64')
    qt = qt.astype('float64')

    # Decompress DCT coefficients into the spatial domain.
    # Deliberately not using tools.decompress_channel here: J-MiPOD's Wiener
    # residual is invariant to a constant level shift in the interior, but the
    # zero-padding boundary used by wiener2 is not, so we must decompress the
    # same way as the reference Matlab implementation (no +128 level shift).
    dequantized_dct_coeffs = y0 * qt[None, None, :, :]
    cover_decompressed = tools.dct.block_idct2(dequantized_dct_coeffs)
    cover_decompressed = tools.dct.jpeglib_to_jpegio(cover_decompressed)

    # Compute wiener residual
    # Suppressed the content, the residual is zero mean
    # The residual still contains remnants of the content around edges and in textured areas.
    wiener_residual = cover_decompressed - _defs.wiener2(cover_decompressed, kernel_size=(2, 2))

    # Estimate variance: Fit a local parameteric model to the neighbors of each pixel value
    variance = estimate_variance(wiener_residual, block_size=3, degree=3)

    # Apply the covariance transformation to the DCT domain.
    # Reshape variance from 2D to 3D, one row of 64 values per 8x8 block.
    num_vertical_blocks = variance.shape[0] // 8
    num_horizontal_blocks = variance.shape[1] // 8
    variance_3d = variance.reshape(
        (num_vertical_blocks, 8, num_horizontal_blocks, 8),
    ).transpose((0, 2, 1, 3)).reshape((num_vertical_blocks, num_horizontal_blocks, 64))

    # Compute DCT basis functions of shape [64, 64]
    dct_basis_functions = tools.dct.compute_dct_basis_functions()

    # x is an array of length 64
    variance_dct_3d = np.apply_along_axis(
        lambda x: np.diag(dct_basis_functions @ np.diag(x) @ dct_basis_functions.T),
        axis=2,
        arr=variance_3d,
    )

    # Divide by the squared quantization table
    variance_dct_3d /= (qt.flatten() ** 2)[None, None, :]

    # Reshape back to 2D
    variance_dct = variance_dct_3d.reshape(
        (num_vertical_blocks, num_horizontal_blocks, 8, 8),
    ).transpose((0, 2, 1, 3)).reshape((num_vertical_blocks * 8, num_horizontal_blocks * 8))

    # Set minimum variance
    # Similar to Eq. 27 of the (spatial) MiPOD paper, but with a different value.
    variance_dct = np.clip(variance_dct, a_min=1e-5, a_max=None)

    # Compute Fisher information (I = 2 / sigma_n ** 4).
    fisher_information = 1 / variance_dct ** 2

    # Pad the Fisher information by copying 8 pixels on all sides
    fisher_information_padded = _defs.pad_copy_borders(fisher_information, ((8, 8), (8, 8)))

    # Smooth Fisher information.
    # Low-pass filtering the Fisher information spreads the high costs of a pixel into its neighborhood, making embedding more conservative.
    # Smoothing also evens out the costs and thus increases the entropy of embedding changes (the payload) in highly textured regions.
    fisher_information = (
        +     fisher_information_padded[:-16, :-16]    # top left
        + 3 * fisher_information_padded[8:-8, :-16]    # center left
        +     fisher_information_padded[16:, :-16]     # bottom left
        + 3 * fisher_information_padded[:-16, 8:-8]    # top center
        + 4 * fisher_information_padded[8:-8, 8:-8]    # center center
        + 3 * fisher_information_padded[16:, 8:-8]     # bottom center
        +     fisher_information_padded[:-16, 16:]     # top right
        + 3 * fisher_information_padded[8:-8, 16:]     # center right
        +     fisher_information_padded[16:, 16:]      # bottom right
    )

    return fisher_information
