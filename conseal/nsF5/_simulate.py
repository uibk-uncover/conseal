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
    cover: np.ndarray,
    stego: np.ndarray,
) -> float:
    """Estimates payload [bpnzac] embedded in the stego with nsF5.

    Args:
        cover (np.ndarray): Cover component.
        stego (np.ndarray): Stego component.
    Returns:
        (float) Estimated embedding rate.
    """
    num_changes = (cover != stego).sum()
    nzAC = tools.dct.nzAC(cover)
    alpha_hat = tools.H(num_changes / nzAC)
    return alpha_hat


def simulate_single_channel(
    cover_dct_coeffs: np.ndarray,
    embedding_rate: float = 1.,
    seed: int = None,
) -> np.ndarray:
    """Simulate embedding into a single channel.

    :param cover_dct_coeffs: array of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: np.ndarray
    :param embedding_rate: embedding rate
    :type embedding_rate: float
    :param seed: random seed for STC simulator
    :type seed: int
    :return: stego DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: np.ndarray
    """

    assert len(cover_dct_coeffs.shape) == 4, "Expected DCT coefficients to have 4 dimensions"
    assert cover_dct_coeffs.shape[2] == cover_dct_coeffs.shape[3] == 8, "Expected blocks of size 8x8"

    # No embedding
    if np.isclose(embedding_rate, 0):
        return cover_dct_coeffs

    # Compute bound on embedding efficiency
    embedding_efficiency = embedding_rate / tools.inv_entropy(embedding_rate)

    # Number of nonzero AC DCT coefficients
    nzAC = tools.dct.nzAC(cover_dct_coeffs)

    if nzAC == 0:
        raise ValueError('There are no non-zero AC coefficients for embedding')

    # Number of changes nsF5 would make on bound
    num_changes = int(np.ceil(embedding_rate * nzAC / embedding_efficiency))

    # Rearrange DCT coefficients to image shape in order to match the Matlab implementation
    num_vertical_blocks, num_horizontal_blocks, _, _ = cover_dct_coeffs.shape

    # Mask of all nonzero DCT coefficients in the image
    changeable = cover_dct_coeffs != 0

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

    # Temporarily save the permutation
    store_permutation = False
    if store_permutation:
        from scipy.io import savemat

        # Create an array to hold the order of coefficients used by Matlab (only count the changeable coefficients)
        indices_matlab = np.zeros((num_vertical_blocks * 8, num_horizontal_blocks * 8), dtype=int)

        # Because Matlab uses column-major order, reorder our coordinates by the y-dimension
        indices_2d_reordered = indices_2d[np.lexsort((indices_2d[:, 0], indices_2d[:, 1]))]

        # Fill index array sequentially
        indices_matlab[indices_2d_reordered[:, 0], indices_2d_reordered[:, 1]] = np.arange(nzAC)

        # Retrieve the permutation that Matlab will apply
        permutation_matlab = indices_matlab[permuted_indices_2d[:, 0], permuted_indices_2d[:, 1]]

        # Store the permutation to file
        savemat(f"/tmp/random_permutation_seed_{seed}_nzAC_{nzAC}.mat", {"permutation": permutation_matlab})

    # Coefficients to be changed
    to_be_changed_2d = permuted_indices_2d[:num_changes]

    # Flatten cover DCT coefficients
    cover_dct_coeffs_2d = cover_dct_coeffs.transpose((0, 2, 1, 3)).reshape((num_vertical_blocks * 8, num_horizontal_blocks * 8))

    # Buffer containing the changes to the cover image
    delta_2d = np.zeros(cover_dct_coeffs_2d.shape, dtype=np.int8)

    # Decrease the absolute value of the coefficients to be changed
    delta_2d[to_be_changed_2d[:, 0], to_be_changed_2d[:, 1]] = -np.sign(cover_dct_coeffs_2d[to_be_changed_2d[:, 0], to_be_changed_2d[:, 1]])

    # Reshape delta to the original shape
    delta = delta_2d.reshape((num_vertical_blocks, 8, num_horizontal_blocks, 8)).transpose((0, 2, 1, 3))

    return cover_dct_coeffs + delta


__all__ = ['simulate_single_channel']
