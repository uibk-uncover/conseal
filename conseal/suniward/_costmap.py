#!/usr/bin/env python3
"""
Implementation of S-UNIWARD steganographic embedding.

Code inspired by Holub's Matlab implementation: https://dde.binghamton.edu/download/stego_algorithms/
Code inspired by https://github.com/daniellerch/stegolab.

Universal Distortion Function for Steganography in an Arbitrary Domain.
V. Holub, J. Fridrich and T. Denemark. http://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf
"""

import numpy as np
from scipy.signal import correlate2d
import typing

from .. import tools


def compute_cost(
    x0: np.ndarray,
    *,
    sigma: float = 1,
) -> np.ndarray:
    """Computes S-UNIWARD cost.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param sigma: stabilizing constant. Small sigmas make the embedding very sensitive to the image content. Large sigmas smooth out the embedding change probabilities.
    :type sigma: float
    :return: cost for +-1 change,
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """  # noqa: E501
    # Get 2D wavelet filters - Daubechies 8
    (high_pass_decomposition_filter, low_pass_decomposition_filter), filters = tools.spatial.daubechies8()
    num_filters = len(filters)

    pad_size = np.max([f.shape for f in filters])

    # Mirror-pad the spatial image. Extend image by the length of the filter in all dimensions.
    # "symmetric" means reflect, i.e. (d c b a | a b c d | d c b a)
    x0_padded = np.pad(x0, (pad_size, pad_size), 'symmetric')

    # Compute directional residual and suitability \xi for each filter
    xis = []
    for filter_idx in range(num_filters):
        # Compute residual

        # The original implementation uses a convolution but we take the correlation with the flipped kernel, which matches the Matlab alignment (otherwise, there is misalignment by 1 pixel).
        residual = correlate2d(x0_padded, np.flipud(np.fliplr(filters[filter_idx])), mode='same', boundary='fill', fillvalue=0)

        # Compute suitability
        xi = correlate2d(1. / (np.abs(residual) + sigma), np.abs(filters[filter_idx]), mode='same', boundary='fill', fillvalue=0)

        # Correct the suitability shift if filter size is even
        if filters[filter_idx].shape[0] % 2 == 0:
            xi = np.roll(xi, 1, axis=0)

        if filters[filter_idx].shape[1] % 2 == 0:
            xi = np.roll(xi, 1, axis=1)

        # Remove padding
        offset_y = (xi.shape[0] - x0.shape[0]) // 2
        offset_x = (xi.shape[1] - x0.shape[1]) // 2

        xi = xi[offset_y:-offset_y, offset_x:-offset_x]

        xis.append(xi)

    # Compute embedding costmap rho
    rho = xis[0] + xis[1] + xis[2]

    return rho


def compute_cost_adjusted(
    x0: np.ndarray,
    *,
    wet_cost: float = 10**8,
    **kw,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Computes S-UNIWARD cost with wet-cost adjustments.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: costs for +1 and -1 changes,
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.suniward.compute_cost_adjusted(x0=x0)
    """
    assert len(x0.shape) == 2, 'single channel expected'

    # process input
    x0 = x0.astype('float64')

    # Calculate costmap
    rho = compute_cost(x0=x0, **kw)

    # Assign wet cost
    rho[np.isinf(rho) | np.isnan(rho) | (rho > wet_cost)] = wet_cost
    # rho[(rho > wet_cost) | np.isnan(rho)] = wet_cost

    # Do not embed +1 if the pixel has max value
    rho_p1 = np.copy(rho)
    rho_p1[x0 == 255] = wet_cost

    # Do not embed -1 if the pixel has min value
    rho_m1 = np.copy(rho)
    rho_m1[x0 == 0] = wet_cost

    # return costs
    return rho_p1, rho_m1


# def compute_cost_per_direction(
#     cover: np.ndarray,
#     sigma: float = 1,
# ) -> np.ndarray:
#     """
#     Produce UNIWARD distortion for given cover. Keep the three directions separate.

#     The first filter evaluates the smoothness in horizontal direction. The filter locates vertical edges.
#     The second filter evaluates the smoothness in vertical direction. The filter locates horizontal edges.
#     The third filter evaluates the smoothness in diagonal direction. The filter locates diagonal edges.

#     If the residual is high in direction d, the cost of embedding in direction d is low.

#     Args:
#         cover (np.ndarray): Cover image in spatial domain.
#         sigma (float): stabilizing constant. Small sigmas make the embedding very sensitive to the image content. Large sigmas smooth out the embedding change probabilities.
#     Returns:
#         costmap (np.ndarray): Same size as the cover image. The distortion for each of the three dimensions is stacked along the last axis.
#             First channel: Horizontal filter -> vertical edges
#             Second channel: Vertical filter -> horizontal edges
#             Third channel: Diagonal filter -> diagonal edges
#     """

#     # Get 2D wavelet filters - Daubechies 8
#     # 1D high-pass decomposition filter
#     # In the paper, the high-pass filter is denoted as g.
#     high_pass_decomposition_filter = np.array([-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768])
#     wavelet_filter_size = len(high_pass_decomposition_filter)

#     # 1D low-pass decomposition filter
#     # In the paper, the low-pass filter is denotes as h.
#     low_pass_decomposition_filter = np.power(-1, np.arange(len(high_pass_decomposition_filter))) * np.flip( high_pass_decomposition_filter)

#     # Stack filter kernels to shape [3, 16, 16]
#     # K^1 = h * g.T
#     # K^2 = g * h.T
#     # K^3 = g * g.T
#     filters = np.stack([
#         low_pass_decomposition_filter[:, None] * high_pass_decomposition_filter[None, :],
#         high_pass_decomposition_filter[:, None] * low_pass_decomposition_filter[None, :],
#         high_pass_decomposition_filter[:, None] * high_pass_decomposition_filter[None, :],
#     ], axis=0)
#     num_filters = len(filters)

#     pad_size = np.max([f.shape for f in filters])

#     # Mirror-pad the spatial image. Extend image by the length of the filter in all dimensions.
#     # "symmetric" means reflect, i.e. (d c b a | a b c d | d c b a)
#     cover_padded = np.pad(cover, (pad_size, pad_size), 'symmetric')

#     # Compute directional residual and suitability \xi for each filter
#     xis = []
#     for filter_idx in range(num_filters):
#         # Compute residual

#         # The original implementation uses a convolution but we take the correlation with the flipped kernel, which matches the Matlab alignment (otherwise, there is misalignment by 1 pixel).
#         residual = correlate2d(cover_padded, np.flipud(np.fliplr(filters[filter_idx])), mode='same', boundary='fill', fillvalue=0)

#         # Compute suitability
#         xi = correlate2d(1. / (np.abs(residual) + sigma), np.abs(filters[filter_idx]), mode='same', boundary='fill', fillvalue=0)

#         # Correct the suitability shift if filter size is even
#         if filters[filter_idx].shape[0] % 2 == 0:
#             xi = np.roll(xi, 1, axis=0)

#         if filters[filter_idx].shape[1] % 2 == 0:
#             xi = np.roll(xi, 1, axis=1)

#         # Remove padding
#         offset_y = (xi.shape[0] - cover.shape[0]) // 2
#         offset_x = (xi.shape[1] - cover.shape[1]) // 2

#         xi = xi[offset_y:-offset_y, offset_x:-offset_x]

#         xis.append(xi)

#     # Compute embedding costmap rho
#     rho_per_direction = np.stack(xis, axis=-1)

#     return rho_per_direction
