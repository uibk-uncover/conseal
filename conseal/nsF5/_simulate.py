"""Implementation of nsF5 steganographic embedding.

Described in:
J. Fridrich, T. Pevny, and J. Kodovsky.
"Statistically undetectable JPEG steganography: Dead ends, challenges, and opportunities,"
MMSec 2007. http://dde.binghamton.edu/kodovsky/pdf/Fri07-ACM.pdf

Code inspired by Kodovsky's Matlab implementation:
https://dde.binghamton.edu/download/stego_algorithms/

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

from collections.abc import Sequence
import numpy as np
from typing import List, Union, Tuple

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


def probability(
    cover_dct_coefs: np.ndarray,
    alpha: float = 1.,
) -> np.ndarray:
    """Returns nsF5 probability map for consequent simulation."""

    assert len(cover_dct_coefs.shape) == 4, "Expected DCT coefficients to have 4 dimensions"
    assert cover_dct_coefs.shape[2] == cover_dct_coefs.shape[3] == 8, "Expected blocks of size 8x8"

    # No embedding
    if np.isclose(alpha, 0):
        return np.zeros_like(cover_dct_coefs)

    # Compute change rate on bound
    beta = tools.inv_entropy(alpha)

    # Number of nonzero AC DCT coefficients
    nzAC = tools.dct.nzAC(cover_dct_coefs)
    if nzAC == 0:
        raise ValueError('There are no non-zero AC coefficients for embedding')

    # probability map
    p = np.ones(cover_dct_coefs.shape, dtype='float64') * beta

    # do not change zeros or DC mode
    p[cover_dct_coefs == 0] = 0
    p[:, :, 0, 0] = 0

    # substract absolute value
    pP1, pM1 = p.copy(), p.copy()
    pP1[cover_dct_coefs > 0] = 0
    pM1[cover_dct_coefs < 0] = 0

    return pP1, pM1


def simulate_single_channel(
    cover_dct_coeffs: np.ndarray,
    embedding_rate: float = 1.,
    seed: int = None,
) -> np.ndarray:

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


def simulate_batch(
    cover_dct_coeffs: Tuple[np.ndarray],
    alphas: Tuple[float],
    seed: int,
) -> List[np.ndarray]:
    """

    Args:
        cover_dct_coefs list of np.ndarrays: Batch of cover DCT coefficients.
        alpha list of floats: Embedding rate for each of the items in the batch
        seed (int): Seed for random generator.
    """
    # Set up buffer for results
    results_buffer = []

    # Iterate over number of input items or channels
    batch_size = len(cover_dct_coeffs)
    for i in range(batch_size):
        # Simulate embedding for single channel
        channel = cover_dct_coeffs[i]
        alpha = alphas[i]

        stego_dct_coeffs = simulate_single_channel(channel, alpha, seed)

        # Append result to buffer
        results_buffer.append(stego_dct_coeffs)

    return results_buffer


def simulate(
    cover_dct_coeffs: Union[np.ndarray, Tuple[np.ndarray]],
    embedding_rate: Union[float, Tuple[float]] = 1.,
    seed: int = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Wrapper method to simulate nsF5 embedding.

    Given a single input DCT coefficients, the method calls `simulate_single_channel`.
    Given a list of DCT coefficients, the method calls `simulate_batch`.

    Returns ndarray with changes to be applied to the input coefficients, or a list of ndarrays in batch mode.

    Args:
        cover_dct_coefs (single np.ndarray, or list of np.ndarrays): Cover DCT(s).
        embedding_rate (single float, or list of floats): Embedding rate.
        seed (int): Seed for random generator.
    """

    is_batch = isinstance(cover_dct_coeffs, Sequence)

    # Process batch
    if is_batch:
        batch_size = len(cover_dct_coeffs)
        alphas = embedding_rate

        # If alpha is a single number, expand alpha to batch shape
        if not isinstance(alphas, Sequence):
            alphas = (embedding_rate,) * batch_size

        return simulate_batch(cover_dct_coeffs=cover_dct_coeffs, alphas=alphas, seed=seed)

    # Process single channel
    return simulate_single_channel(
        cover_dct_coeffs=cover_dct_coeffs,
        embedding_rate=embedding_rate,
        seed=seed,
    )


__all__ = ['simulate']
