"""
Implementation of the nsF5 steganography method as described in

J. Fridrich, T. Pevny, and J. Kodovsky.
"Statistically undetectable JPEG steganography: Dead ends, challenges, and opportunities"
Multimedia & Security, 2007
http://dde.binghamton.edu/kodovsky/pdf/Fri07-ACM.pdf

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np

from .. import tools


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
