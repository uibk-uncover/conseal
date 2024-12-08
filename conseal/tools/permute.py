""""""

import hashlib
import numpy as np
import typing


def blocks(
    y: np.ndarray,
    key: int = None,
) -> typing.Tuple[int]:
    """Returns indices of permuted coefficients."""

    # permute
    rng = np.random.default_rng(key)
    perm = rng.permutation(y.size)

    # dimensions
    Hb, Wb, _, _ = y.shape
    wb = perm % 8
    hb = perm // 8 % 8
    w = perm // 64 % Wb
    h = perm // 64 // Wb
    return h, w, hb, wb


def iterate_ac(
    dct: np.ndarray,
    key: int,
    skipzero: bool = True
):
    """Generator for iterating ACs/nzACs."""
    # permutative straddling
    perm = blocks(dct, key)

    # iterate coefficients
    for cover_it in range(dct.size):

        # get cover element
        h, w, hb, wb = [pj[cover_it] for pj in perm]

        # skip
        if hb == 0 and wb == 0:  # DC
            continue
        if skipzero and dct[h, w, hb, wb] == 0:  # zero
            continue

        # append to window
        yield h, w, hb, wb


def iterate(
    x: np.ndarray,
    key: int = None,
):
    # flip
    flip = len(x.shape) == 4

    # shuffle
    perm = np.linspace(0, x.size-1, x.size, dtype='int64')
    if key:
        rng = np.random.default_rng(key)
        perm = rng.permutation(x.size)

    # prepare shapes
    mod = np.array(x.shape)
    if flip:
        mod = np.flip(mod)
    denom = np.hstack([[1], np.cumprod(mod)[:-1]])

    # iterate elements
    for it in perm:

        # get index
        idx = tuple([
            it // denom[i] % mod[i]
            for i in range(len(x.shape))
        ])

        if flip:
            idx = np.flip(idx)

        yield idx


def password_to_seed(password: str) -> int:
    """Converts password to seed for random generator.

    :param password: string password
    :type password: str
    :param seed: random seed for embedding simulator
    :type seed: int

    :Example:

    >>> seed = cl.tools.password_to_seed(cover_name)  # cover-specific seed
    >>> rng = np.random.default_rng(seed)
    """
    # compute SHA-256 digest of the password
    digest = hashlib.sha256(password.encode('utf-8')).hexdigest()
    # convert to seed
    seed = int(digest, base=16)
    return seed
