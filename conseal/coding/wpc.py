"""Implementation of wet-paper codes, as described in

J. Fridrich, M. Goljan, and D. Soukal.
"Efficient Wet Paper Codes"
Information Hiding, 2005

J. Fridrich, M. Goljan, and D. Soukal.
"Wet Paper Codes with Improved Embedding Efficiency"
IEEE Transactions on Information Forensics and Security, 2006

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import warnings

from .. import tools
from .._conseal import wpc as rs
from . import _lt
from . import _symbol


def generate_H(
    m: int,
    n: int,
    *,
    q: int = 2,
    seed: int = None,
) -> np.ndarray:
    """Generates random parity-check matrix over GF(q) with distinct columns.

    Columns are nonzero, with the leading nonzero digit 1,
    so no column is a scalar multiple of another.
    If n exceeds the (q^m - 1) / (q - 1) such columns, they repeat.

    :param m: number of rows, i.e., message symbols per block
    :type m: int
    :param n: number of columns, i.e., cover elements per block
    :type n: int
    :param q: prime field size, binary by default
    :type q: int
    :param seed: random seed, shared by sender and receiver
    :type seed: int
    :return: generated parity-check matrix of shape (m, n)
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> H = generate_H(18, 200, seed=12345)
    """
    # normalized nonzero columns, leading nonzero digit is 1
    columns = np.arange(1, q**m)
    digits = _lt._decode(columns, q, m)
    leading = digits[(digits != 0).argmax(axis=0), np.arange(len(columns))]
    columns = columns[leading == 1]

    # distinct, if possible
    rng = np.random.default_rng(seed)
    columns = rng.choice(columns, size=n, replace=n > len(columns))
    return _lt._decode(columns, q, m)


def _block_size(q: int, block_size: int = None) -> int:
    """Checks block size p, by default the largest with q^p <= 2^18."""
    if block_size is None:
        block_size = int(18 / np.log2(q) + 1e-9)
    if q**block_size > 2**22:
        raise ValueError(f'block_size {block_size} too large, q^block_size must be <= 2^22')
    return block_size


def _blocks(
    n: int,
    m: int,
    block_size: int,
) -> list:
    """Splits cover elements and message symbols into contiguous blocks.

    :return: blocks, each a tuple (elements, symbols)
        of contiguous element indices and symbol indices,
        sizes differ by at most 1
    :rtype: list
    """
    # number of blocks, rounded up
    num_blocks = -(-m // block_size)

    # split elements and symbols evenly
    return list(zip(
        np.array_split(np.arange(n), num_blocks),
        np.array_split(np.arange(m), num_blocks),
    ))


def _choose_backend(q: int) -> tools.Backend:
    """Backend, Python if Rust does not support q."""
    backend = tools.get_backend()
    if backend == tools.BACKEND_RUST and q >= 256:
        warnings.warn(
            f'Rust backend of WPC requires q < 256, got {q=}, '
            'falling back to Python',
            RuntimeWarning,
        )
        backend = tools.BACKEND_PYTHON
    return backend


def _check_symbols(x: np.ndarray, q: int):
    """Raises ValueError, if x are not GF(q) symbols."""
    if ((x < 0) | (x >= q)).any():
        raise ValueError(f'expected symbols in 0..{q-1}')


def embed(
    x: np.ndarray,
    rhos: np.ndarray,
    message: bytes,
    *,
    seed: int = 0,
    add_zero: bool = True,
    wet_cost: float = 10**10,
    block_size: int = None,
) -> np.ndarray:
    """Embeds message into cover symbols with wet-paper code.

    Returns which cover symbols to change, so that the receiver
    reads the message without knowing which elements were wet.
    Wet elements never change, and among the dry ones,
    the code makes as few changes as possible.
    Applying the changes to the cover is up to the caller.

    Embedding fails, if the message is too long for the dry elements,
    for binary reliably below 0.3 bits per dry element.

    :param x: cover symbols in 0..q-1, in random order
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rhos: costs of the changes, one tensor per change,
        change k shifts the symbol by k mod q, e.g.,
        (rho,) for binary, (rho_p1, rho_m1) for ternary
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param message: message to embed
    :type message: bytes
    :param seed: seed of the parity-check matrix, shared with the receiver,
        0 by default
    :type seed: int
    :param add_zero: whether no change is implicit with zero cost,
        as in :func:`conseal.simulator.get_probability`
    :type add_zero: bool
    :param wet_cost: costs from this value up are wet, 10^10 by default
    :type wet_cost: float
    :param block_size: number of message symbols per block p,
        by default the largest with q^p <= 2^18,
        i.e., 18 for binary and 11 for ternary,
        at most q^p <= 2^22
    :type block_size: int
    :return: changes k in 0..q-1, of the shape of x, 0 for no change
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :raises ValueError: if the message does not fit into some block,
        or if no change is wet for some element

    :Example:

    >>> # nsF5: parity of the coefficients, decreased in absolute value
    >>> order = np.random.default_rng(12345).permutation(y0.size)  # keyed order
    >>> k = np.zeros(y0.size, dtype='uint8')
    >>> k[order] = cl.coding.wpc.embed(
    ...     (y0 % 2).flatten()[order],    # cover parities
    ...     (rho.flatten()[order],),      # costs of the parity flip
    ...     b'secret')                    # message
    >>> y1 = y0 - np.sign(y0) * k.reshape(y0.shape)
    """
    # flattened cover symbols
    x = np.asarray(x)
    n = x.size
    x_flat = x.reshape(-1).astype('int64')

    # one row of costs per change, no change implicit or first
    rhos = np.asarray(rhos, dtype=float).reshape(-1, n)
    if add_zero:
        rho_zero = np.zeros(n)
    else:
        rho_zero, rhos = rhos[0], rhos[1:]
        # code cannot force a change, solver decides
        if not (rho_zero < wet_cost).all():
            raise ValueError('no change must not be wet')
    q = len(rhos) + 1
    _check_symbols(x, q)
    block_size = _block_size(q, block_size)

    # message as GF(q) symbols, nothing to embed for empty message
    symbols = _symbol.bytes_to_symbols(message, q)
    m = len(symbols)
    changes = np.zeros(n, dtype='uint8')
    if m == 0:
        return changes.reshape(x.shape)

    # dry, if all changes cost below wet_cost,
    # since solver may require any of them
    # (inf and nan compare False, so they are wet too)
    is_dry = (rhos < wet_cost).all(axis=0)

    # cost of a change relative to no change, averaged over the changes
    rho = rhos.mean(axis=0) - rho_zero

    # blocks and their common parity-check matrix
    blocks = _blocks(n, m, block_size)
    H_max = generate_H(
        max(len(block_symbols) for _, block_symbols in blocks),
        max(len(elements) for elements, _ in blocks),
        q=q, seed=seed,
    )

    # choose implementation
    backend = _choose_backend(q)
    if backend == tools.BACKEND_RUST:
        changes = rs.embed(
            x_flat.astype('uint8'), is_dry, rho.astype('float64'),
            symbols.astype('uint8'), np.ascontiguousarray(H_max),
            block_size, q,
        )
    elif backend == tools.BACKEND_PYTHON:
        changes = _embed(x_flat, is_dry, rho, symbols, H_max, blocks, q)
    else:
        raise NotImplementedError(f'unknown backend {backend}')

    # changes, in the cover shape
    return changes.reshape(x.shape)


def _embed(
    x: np.ndarray,
    is_dry: np.ndarray,
    rho: np.ndarray,
    symbols: np.ndarray,
    H_max: np.ndarray,
    blocks: list,
    q: int,
) -> np.ndarray:
    """Python implementation of the block loop of :func:`embed`."""
    changes = np.zeros(len(x), dtype='uint8')
    for i, (elements, block_symbols) in enumerate(blocks):

        # dry columns of the block, cheaper first,
        # solver uses the first of equal columns
        dry = np.flatnonzero(is_dry[elements])
        dry = dry[np.argsort(rho[elements[dry]], kind='stable')]

        # common parity-check matrix, sliced to the block
        H = H_max[:len(block_symbols), :len(elements)]

        # syndrome difference z = message - H x,
        # to be produced by changing only dry elements, D v = z
        z = (symbols[block_symbols] - H @ x[elements]) % q
        try:
            v = _lt.lt_solve(H[:, dry], z, q=q)
        except ValueError as e:
            raise ValueError(
                f'block {i}: {len(block_symbols)} symbols do not fit '
                f'into {len(dry)} dry elements'
            ) from e

        # symbol shifts are the changes, back in cover order
        changes[elements[dry]] = v
    return changes


def extract(
    y: np.ndarray,
    length: int,
    *,
    seed: int = 0,
    q: int = 2,
    block_size: int = None,
) -> bytes:
    """Extracts message from stego symbols, embedded with wet-paper code.

    Computes the syndrome of each block,
    with the same blocks and parity-check matrix as :func:`embed`.

    :param y: stego symbols in 0..q-1, in the order used by :func:`embed`
    :type y: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param length: message length in bytes
    :type length: int
    :param seed: seed of the parity-check matrix, as in :func:`embed`
    :type seed: int
    :param q: prime field size, i.e., number of symbols,
        len(rhos) + 1 in :func:`embed`, binary by default
    :type q: int
    :param block_size: number of message symbols per block,
        as in :func:`embed`
    :type block_size: int
    :return: extracted message
    :rtype: bytes

    :Example:

    >>> order = np.random.default_rng(12345).permutation(y1.size)  # as sender
    >>> message = cl.coding.wpc.extract((y1 % 2).flatten()[order], 6)
    """
    # number of message symbols, nothing to extract for empty message
    m = _symbol.num_symbols(length, q)
    if m == 0:
        return b''

    # flattened stego symbols
    y = np.asarray(y)
    _check_symbols(y, q)
    block_size = _block_size(q, block_size)
    y_flat = y.reshape(-1).astype('int64')

    # same blocks and parity-check matrix as the sender
    blocks = _blocks(y_flat.size, m, block_size)
    H_max = generate_H(
        max(len(block_symbols) for _, block_symbols in blocks),
        max(len(elements) for elements, _ in blocks),
        q=q, seed=seed,
    )

    # choose implementation
    backend = _choose_backend(q)
    if backend == tools.BACKEND_RUST:
        symbols = rs.extract(
            y_flat.astype('uint8'), m, np.ascontiguousarray(H_max),
            block_size, q,
        )
    elif backend == tools.BACKEND_PYTHON:
        symbols = _extract(y_flat, m, H_max, blocks, q)
    else:
        raise NotImplementedError(f'unknown backend {backend}')

    # symbols back to bytes
    return _symbol.symbols_to_bytes(symbols, length, q)


def _extract(
    y: np.ndarray,
    m: int,
    H_max: np.ndarray,
    blocks: list,
    q: int,
) -> np.ndarray:
    """Python implementation of the block loop of :func:`extract`."""
    # message is the syndrome H y of each block,
    # no need to know which elements were wet
    symbols = np.zeros(m, dtype='int64')
    for elements, block_symbols in blocks:
        H = H_max[:len(block_symbols), :len(elements)]
        symbols[block_symbols] = H @ y[elements] % q
    return symbols
