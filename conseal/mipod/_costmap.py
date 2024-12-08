"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.signal
# from scipy.signal import correlate2d
from . import _defs
from .. import tools
# from stegolab2.jmipod.costmap import wiener2, estimate_variance


def estimate_variance(
    image: np.ndarray,
    block_size: int,
    degree: int,
) -> np.ndarray:
    """
    Estimate the pixels' variance by fitting a local parametric model based on a 2D-DCT (trigonometric polynomial) to the neighbors of each residual value.
    :param image: image residual
    :type image: np.ndarray
    :param block_size: size of the local neighborhood, denoted as p in the paper. See Fig. 2.
        Small p: extreme content adaptivity;
        Medium p: Medium content adaptivity;
        Large p: Low content adaptivity.
    :type block_size: int
    :param degree: degree of the polynomial
    :type degree: int
    :return: estimated variance per pixel
    :rtype:
    """
    assert block_size % 2 != 0, 'block dimensions must be odd'
    assert degree <= block_size, 'number of basis vectors exceeds block size'

    # Number of parameters per p x p block, see sentence below Eq. 26
    q = degree * (degree + 1) // 2

    base_mat = np.zeros((block_size, block_size))
    base_mat[0, 0] = 1

    # Construct G matrix of size [p * p, q]. G defines the parametric model with 2-D discrete trigonometrical polynomial functions. See Eq. (25) and Eq. (26).
    G = np.zeros((block_size ** 2, q))
    k = 0
    for xshift in range(degree):
        for yshift in range(degree - xshift):
            base_mat_rolled = np.roll(base_mat, shift=(xshift, yshift), axis=(0, 1))
            G[:, k] = tools.dct.idct2(base_mat_rolled).flatten(order="F")
            k += 1

    # Symmetry-pad image by block_size / 2 in all directions
    pad_size = block_size // 2
    img_padded = np.pad(image, (pad_size, pad_size), "symmetric")

    # Solve G.T G x = G.T for x.
    # x = (G.T G)^{-1} G.T
    x = np.linalg.solve(G.T @ G, G.T)

    # P_G_orth = I - G x = I - G (G^T G)^{-1} G.T
    # P_G_orth represents the orthogonal projection ontho the p^2 - q dimensional subspace spanned by the left null space of G.
    P_G_orth = np.eye(block_size ** 2) - (G @ x)

    sliding_window_views = _defs.im2col(
        img_padded,
        window_shape=(block_size, block_size),
    )

    # Eq. 24
    estimated_variance = np.sum((P_G_orth @ sliding_window_views) ** 2, axis=0) / (block_size ** 2 - q)
    estimated_variance = estimated_variance.reshape(image.shape)

    # The estimated value of the variance is attributed to the center pixel.
    return estimated_variance


def compute_cost(x0: np.ndarray) -> np.ndarray:
    """Produces MiPOD cost. Not a cost like HILL, but a Fisher information.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: cost for +-1 change
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> fi = cl.mipod.compute_cost(x0=x0)
    """
    x0 = x0.astype(float)

    # Compute wiener residual
    # Suppressed the content, the residual is zero mean
    # The residual still contains remnants of the content around edges and in textured areas.
    wiener_residual = x0 - _defs.wiener2(x0, kernel_size=(2, 2))

    # Estimate variance: Fit a local parameteric model to the neighbors of each pixel value
    variance = estimate_variance(wiener_residual, block_size=9, degree=9)

    # Set minimum variance, Eq. 27
    variance = np.clip(variance, a_min=0.01, a_max=None)

    # Compute Fisher information (I = 2 / sigma_n ** 4).
    fisher_information = 1 / variance ** 2

    # average_kernel =
    fisher_information = scipy.signal.correlate2d(
        fisher_information,
        np.ones((7, 7)) / 7 ** 2,
        mode="same", boundary="symm",
    )

    return fisher_information
