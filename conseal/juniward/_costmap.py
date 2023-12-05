"""
Implementation of the J-UNIWARD steganography method as described in

V. Holub, J. Fridrich, T. Denemark
"Universal distortion function for steganography in an arbitrary domain"
EURASIP Journal on Information Security, 2014
http://www.ws.binghamton.edu/fridrich/research/uniward-eurasip-final.pdf

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck

This implementation builds on the original Matlab implementation provided by the paper authors. Please find that license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2013 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501

import enum
import numpy as np
from scipy.signal import correlate2d
import typing

from .. import tools


class Implementation(enum.Enum):
    """J-UNIWARD implementation to choose from."""

    JUNIWARD_ORIGINAL = enum.auto()
    """Original J-UNIWARD implementation."""
    JUNIWARD_FIX_OFF_BY_ONE = enum.auto()
    """J-UNIWARD implementation with fixed off-by-one error.

    See https://arxiv.org/pdf/2305.19776.pdf for more details.
    """


def compute_cost(
    spatial: np.ndarray,
    quantization_table: np.ndarray,
    dtype: np.dtype = np.float64,
    implementation: Implementation = Implementation.JUNIWARD_ORIGINAL,
) -> np.ndarray:
    """Compute the UNIWARD distortion function for a given JPEG cover image.

    :param spatial: grayscale image in spatial domain
    :type spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param quantization_table: quantization table of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param dtype: float32 or float64
    :type dtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.dtype.html>`_
    :param implementation: choose J-UNIWARD implementation
    :type implementation: :class:`Implementation`
    :return: cost map for embedding into the DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_

    :Example:

    >>> # TODO
    """
    assert len(spatial.shape) == 2, "Expected grayscale image"
    height, width = spatial.shape
    assert height % 8 == 0, "Expected height to be a multiple of 8"
    assert width % 8 == 0, "Expected width to be a multiple of 8"

    # Sigma avoids division by zero but also controls the content sensitivity.
    # Very small sigmas make the embedding very sensitive to the image content.
    # Large sigmas smooth out the embedding change probabilities.
    sigma = 2 ** (-6)

    # Get 2D wavelet filters - Daubechies 8
    # 1D high-pass decomposition filter
    # In the paper, the high-pass filter is denoted as g.
    high_pass_decomposition_filter = np.array([-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768])
    wavelet_filter_size = len(high_pass_decomposition_filter)

    # 1D low-pass decomposition filter
    # In the paper, the low-pass filter is denotes as h.
    low_pass_decomposition_filter = np.power(-1, np.arange(len(high_pass_decomposition_filter))) * np.flip(high_pass_decomposition_filter)

    # Stack filter kernels to shape [3, 16, 16]
    # K^1 = h * g.T
    # K^2 = g * h.T
    # K^3 = g * g.T
    filters = np.stack([
        low_pass_decomposition_filter[:, None] * high_pass_decomposition_filter[None, :],
        high_pass_decomposition_filter[:, None] * low_pass_decomposition_filter[None, :],
        high_pass_decomposition_filter[:, None] * high_pass_decomposition_filter[None, :],
    ], axis=0)
    num_filters = len(filters)

    # Pre-compute impact in spatial domain when a JPEG coefficient is changed by 1
    spatial_impact = np.zeros((8, 8, 8, 8), dtype=dtype)
    for j in range(8):
        for i in range(8):
            # Simulate a single pixel change in the DCT domain
            test_coeffs = np.zeros((8, 8), dtype=dtype)
            test_coeffs[j, i] = 1
            spatial_impact[j, i] = tools.dct.idct2(test_coeffs) * quantization_table[j, i]

    # Pre-compute impact on wavelet coefficients when a JPEG coefficients is changed by 1
    # Changing one pixel will affect s x s wavelet coefficients, where s is the size of the 2D wavelet support.
    # Changing a DCT coefficient will affect a block of 8x8 pixels and therefore a block of (8 + s - 1) x (8 + s - 1) wavelet coefficients.
    change_window_len = 8 + wavelet_filter_size - 1
    assert change_window_len == 23, "Expected change window in Wavelet domain to have length 23"
    wavelet_impact = np.zeros((num_filters, 8, 8, change_window_len, change_window_len), dtype=dtype)
    for filter_idx in range(num_filters):
        for j in range(8):
            for i in range(8):
                # Output length of mode='full': N + M - 1
                wavelet_impact[filter_idx, j, i, :, :] = correlate2d(spatial_impact[j, i], filters[filter_idx], mode='full', boundary='fill', fillvalue=0)

    # Take absolute value, needed for later
    wavelet_impact = np.abs(wavelet_impact)

    # Create reference cover wavelet coefficients (LH, HL, HH)
    # Embedding should minimize their relative change.
    # Computation uses mirror-padding.
    pad_size = len(high_pass_decomposition_filter)

    # Mirror-pad the spatial image. Extend image by the length of the filter in all dimensions.
    # Example: Image [512, 512] -> [16 + 512 + 16, 16 + 512 + 16] = [544, 544]
    # "symmetric" means reflect, i.e. (d c b a | a b c d | d c b a)
    spatial_padded = np.pad(spatial, (pad_size, pad_size), mode='symmetric')

    # Compute the wavelet coefficients of the cover image
    reference_covers = []
    for filter_idx in range(num_filters):
        # Compute wavelet coefficients in the (filter_idx)-th subband
        # The resulting shape is [544, 544]
        rc = correlate2d(spatial_padded, filters[filter_idx], mode='same', boundary='fill', fillvalue=0)

        # Crop as needed
        if implementation == Implementation.JUNIWARD_ORIGINAL:
            rc = rc[pad_size - 7:height + 8 + pad_size, pad_size - 7:width + 8 + pad_size]

        # My proposal
        elif implementation == Implementation.JUNIWARD_FIX_OFF_BY_ONE:
            # Can also be verified by setting the DCT coefficient to a very high value.
            rc = rc[pad_size - 8: height + 7 + pad_size, pad_size - 8:width + 7 + pad_size]

        reference_covers.append(rc)

    #
    # Computation of costs
    #
    window_shape = (23, 23)
    stride = (8, 8)

    arr = reference_covers[0]
    # Compute the shape of the output array
    out_shape = ((arr.shape[0] - window_shape[0]) // stride[0] + 1,
                 (arr.shape[1] - window_shape[1]) // stride[1] + 1,
                 window_shape[0],
                 window_shape[1])

    # Compute the strides of the output array
    out_strides = (arr.strides[0] * stride[0],
                   arr.strides[1] * stride[1],
                   arr.strides[0],
                   arr.strides[1])

    # Use as_strided to create the sliding window view of the Wavelet coefficients
    arr_views = [np.lib.stride_tricks.as_strided(rc, shape=out_shape, strides=out_strides) for rc in reference_covers]

    # Prepare division by the Wavelet coefficients
    reciprocal_views = [np.reciprocal(np.abs(av) + sigma) for av in arr_views]

    # Divide over the [23, 23] subwindows without creating temporary arrays
    temp_xi_0 = np.einsum('ijkl,mnkl->mnij', wavelet_impact[0], reciprocal_views[0])
    temp_xi_1 = np.einsum('ijkl,mnkl->mnij', wavelet_impact[1], reciprocal_views[1])
    temp_xi_2 = np.einsum('ijkl,mnkl->mnij', wavelet_impact[2], reciprocal_views[2])

    # Sum over the three filters
    costs = temp_xi_0 + temp_xi_1 + temp_xi_2

    return costs


def compute_distortion(
    cover_spatial: np.ndarray,
    cover_dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
    dtype: typing.Type = np.float64,
    implementation: Implementation = Implementation.JUNIWARD_ORIGINAL,
    wet_cost: float = 10**13,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Computes the distortion rho_p1 and rho_m1.

    :param cover_spatial: decompressed DCT, array of shape [8*num_vertical_blocks, 8*num_horizontal_blocks]
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param cover_dct_coeffs: array of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param quantization_table: ndarray of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param dtype: float32 or float64
    :type dtype: np.dtype
    :param implementation: choose J-UNIWARD implementation
    :type implementation: :class:`Implementation`
    :param wet_cost: cost for unembeddable coefficients
    :type wet_cost: float
    :return: probability maps for +1 and -1 changes,
        in DCT domain of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: tuple

    :Example:

    >>> # TODO
    """
    # Count number of embeddable DCT coefficients
    num_non_zero_AC_coeffs = tools.dct.nzAC(cover_dct_coeffs)

    if num_non_zero_AC_coeffs == 0:
        raise ValueError('Expected non-zero AC coefficients')

    # Compute costmap
    costmap = compute_cost(
        spatial=cover_spatial,
        quantization_table=quantization_table,
        dtype=dtype,
        implementation=implementation,
    )

    # Assign wet cost
    costmap[
        np.isinf(costmap) |
        np.isnan(costmap) |
        (costmap > wet_cost)] = wet_cost

    # Do not embed +1 if the DCT coefficient has max value
    rho_p1 = np.copy(costmap)
    rho_p1[cover_dct_coeffs >= 1023] = wet_cost

    # Do not embed -1 if the DCT coefficient has min value
    rho_m1 = np.copy(costmap)
    rho_m1[cover_dct_coeffs <= -1023] = wet_cost

    return rho_p1, rho_m1
