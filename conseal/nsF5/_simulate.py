"""
Implementation of the nsF5 steganography method as described in

J. Fridrich, T. Pevny, and J. Kodovsky.
"Statistically undetectable JPEG steganography: Dead ends, challenges, and opportunities"
Multimedia & Security, 2007
http://dde.binghamton.edu/kodovsky/pdf/Fri07-ACM.pdf

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck

This implementation is derived from the original Matlab implementation provided by the paper authors: https://dde.binghamton.edu/download/stego_algorithms/
"""

import numpy as np

from .. import tools


def average_payload(
    y0: np.ndarray,
    y1: np.ndarray,
) -> float:
    """Estimates payload [bpnzac] embedded in the stego with nsF5.

    :param y0: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param y1: quantized stego DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: estimated embedding rate
    :rtype: float

    :Example:

    >>> # TODO
    """
    num_changes = (y0 != y1).sum()
    nzAC = tools.dct.nzAC(y0)
    return tools.H(num_changes / nzAC)


def simulate_single_channel(
    y0: np.ndarray,
    alpha: float,
    *,
    add_shrinkage: bool = False,
    seed: int = None,
) -> np.ndarray:
    """Simulates nsF5 embedding at an embedding rate into single-channel cover and returns stego.

    nsF5 was introduced in
    J. Fridrich, et al. Statistically undetectable JPEG steganography: Dead ends, challenges, and opportunities.
    ACM MMSec, 2007.

    The details of the methods are described in the
    `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#non-shrinkage-f5-nsf5>`__.

    :param y0: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
    :type alpha: float
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: quantized stego DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y = cl.nsF5.simulate_single_channel(
    ...   y0=im_dct.Y,  # DCT
    ...   alpha=0.4,  # alpha
    ...   seed=12345)  # seed
    """
    assert len(y0.shape) == 4, "Expected DCT coefficients to have 4 dimensions"
    assert y0.shape[2] == y0.shape[3] == 8, "Expected blocks of size 8x8"

    # No embedding
    if np.isclose(alpha, 0):
        return y0

    # Compute bound on embedding efficiency
    e = alpha / tools.inv_entropy(alpha)

    # Number of nonzero DCT AC coefficients
    nzAC = tools.dct.nzAC(y0)
    assert nzAC > 0, 'there are no coefficients to embed to'

    # Calculate change rate on the bound
    beta = alpha / e

    # # Introduce shrinkage
    # if add_shrinkage:
    #     p_pm1 = np.mean(np.abs(y0) == 1)  # probability of 1 or -1
    #     beta = beta / (1 - p_pm1)  # adjust change rate
    #     """
    #     beta' = beta + beta * p + beta * p**2 + ...
    #     beta' = beta + beta * (sum_{i=1:\infty} p**i)
    #     beta' = beta + beta * p / (1 - p)
    #     beta' = beta / (1 - p)
    #     """

    # Number of changes
    num_changes = int(np.ceil(beta * nzAC))

    # # Number of changes nsF5 would make on bound
    # num_changes = int(np.ceil(alpha * nzAC / e))

    # Rearrange DCT coefficients to image shape in order to match the Matlab implementation
    num_vertical_blocks, num_horizontal_blocks, _, _ = y0.shape

    # Mask of all nonzero DCT coefficients in the image
    changeable = y0 != 0

    # Do not embed into DC modes
    changeable[:, :, 0, 0] = False

    # Convert to 2D
    changeable_2d = changeable.transpose((0, 2, 1, 3)).reshape((num_vertical_blocks * 8, num_horizontal_blocks * 8))

    # Indexes of changeable coefficients
    indices_2d = np.stack(np.where(changeable_2d), axis=1)

    # Permutative straddling
    # Initialize PRNG using given seed
    rng = np.random.RandomState(seed)

    # Create a pseudorandom walk over nonzero AC coefficients
    permutation = rng.permutation(nzAC)

    # Permute indices
    permuted_indices_2d = indices_2d[permutation]

    # # Temporarily save the permutation
    # store_permutation = False
    # if store_permutation:
    #     from scipy.io import savemat

    #     # Create an array to hold the order of coefficients used by Matlab (only count the changeable coefficients)
    #     indices_matlab = np.zeros((num_vertical_blocks * 8, num_horizontal_blocks * 8), dtype=int)

    #     # Because Matlab uses column-major order, reorder our coordinates by the y-dimension
    #     indices_2d_reordered = indices_2d[np.lexsort((indices_2d[:, 0], indices_2d[:, 1]))]

    #     # Fill index array sequentially
    #     indices_matlab[indices_2d_reordered[:, 0], indices_2d_reordered[:, 1]] = np.arange(nzAC)

    #     # Retrieve the permutation that Matlab will apply
    #     permutation_matlab = indices_matlab[permuted_indices_2d[:, 0], permuted_indices_2d[:, 1]]

    #     # Store the permutation to file
    #     savemat(f"/tmp/random_permutation_seed_{seed}_nzAC_{nzAC}.mat", {"permutation": permutation_matlab})

    # Flatten cover DCT coefficients
    y0_2d = y0.transpose((0, 2, 1, 3)).reshape((num_vertical_blocks * 8, num_horizontal_blocks * 8))

    # Re-embed zeros (shrinkage)
    if add_shrinkage:
        start = 0
        num_zeros = (
            np.abs(y0_2d[tuple(permuted_indices_2d[start:num_changes].T)]) == 1
        ).sum()
        while num_zeros > 0:
            # Move segment
            start = num_changes
            num_changes += num_zeros
            # Number of zeros in new segment
            num_zeros = (
                np.abs(y0_2d[tuple(permuted_indices_2d[start:num_changes].T)]) == 1
            ).sum()

    # Coefficients to be changed
    to_be_changed_2d = permuted_indices_2d[:num_changes]

    # Buffer containing the changes to the cover image
    delta_2d = np.zeros(y0_2d.shape, dtype=np.int8)

    # Decrease the absolute value of the coefficients to be changed
    delta_2d[to_be_changed_2d[:, 0], to_be_changed_2d[:, 1]] = -np.sign(y0_2d[to_be_changed_2d[:, 0], to_be_changed_2d[:, 1]])

    # Reshape delta to the original shape
    delta = delta_2d.reshape((num_vertical_blocks, 8, num_horizontal_blocks, 8)).transpose((0, 2, 1, 3))
    # return delta
    return y0 + delta


__all__ = ['simulate_single_channel']
