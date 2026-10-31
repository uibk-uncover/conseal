"""LT solver of wet-paper codes, over prime fields GF(q).

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import functools
import numpy as np
import warnings

from .. import tools
from .._conseal import lt as rs


def _check_prime(q: int):
    """Raises ValueError, if q is not prime."""
    if q < 2 or any(q % d == 0 for d in range(2, int(q**.5) + 1)):
        raise ValueError(f'GF(q) implemented only for prime q, got {q=}')


@functools.lru_cache(maxsize=None)
def _tables(q: int) -> tuple:
    """Lookup tables for digit-wise arithmetic on base-q integers.

    Integers are processed in limbs of L digits, Q = q^L <= 1024 values.

    :return: tuple (L, Q, add, neg, mul) of limb length, limb values,
        and tables of limb-wise addition, negation and scalar multiplication
    :rtype: tuple
    """
    L = max(1, int(np.log(1024) / np.log(q) + 1e-9))
    Q = q**L
    digits = ((np.arange(Q)[:, None] // q**np.arange(L)) % q).astype('uint8')
    weights = q**np.arange(L)
    add = ((digits[:, None, :] + digits[None, :, :]) % q) @ weights
    neg = ((q - digits) % q) @ weights
    mul = np.stack([((a * digits) % q) @ weights for a in range(q)])
    return L, Q, add, neg, mul


def _limbwise(f, q: int, p: int, *xs):
    """Applies limb lookup f to base-q integers xs of p digits."""
    L, Q = _tables(q)[:2]
    result = 0
    for k in range(-(-p // L)):
        base = Q**k
        result = result + f(*[(x // base) % Q for x in xs]) * base
    return result


def _add(a, b, q: int, p: int):
    """Digit-wise a + b mod q, of base-q integers of p digits."""
    table = _tables(q)[2]
    return _limbwise(lambda x, y: table[x, y], q, p, a, b)


def _neg(a, q: int, p: int):
    """Digit-wise -a mod q, of base-q integers of p digits."""
    table = _tables(q)[3]
    return _limbwise(lambda x: table[x], q, p, a)


def _mul(c: int, a, q: int, p: int):
    """Digit-wise c * a mod q, for scalar c, of base-q integers of p digits."""
    table = _tables(q)[4]
    return _limbwise(lambda x: table[c, x], q, p, a)


def _encode(D: np.ndarray, q: int) -> np.ndarray:
    """Columns of digit matrix D as base-q integers."""
    return (D.astype('int64') * q**np.arange(len(D))[:, None]).sum(axis=0)


def _decode(a: np.ndarray, q: int, m: int) -> np.ndarray:
    """Base-q integers as columns of digit matrix with m rows."""
    return ((a[None, :] // q**np.arange(m)[:, None]) % q).astype('uint8')


def lt_solve(
    D: np.ndarray,
    z: np.ndarray,
    *,
    q: int = 2,
) -> np.ndarray:
    """Solves Dv = z over GF(q) with the fewest nonzeros in v.

    Block Minimal method, Fridrich et al., Efficient Wet Paper Codes, IH 2005,
    with the meet-in-the-middle search for the coset leader,
    Algorithm 1 in Fridrich et al.,
    Wet Paper Codes with Improved Embedding Efficiency.
    Each column of D can be added with coefficient 1..q-1.
    Takes time and memory of order q^m, for D with m rows,
    so meant for small m, q^m <= 2^22.

    Among equal columns, the first is used,
    so sort columns by preference to steer the changes.

    Rust backend requires q < 256,
    otherwise falls back to Python with a warning.
    Both backends give the same solution.

    :param D: matrix of shape (m, n) over GF(q)
    :type D: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param z: right-hand side of shape (m,) over GF(q)
    :type z: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param q: prime field size, binary by default
    :type q: int
    :return: solution v of shape (n,)
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :raises ValueError: if the system has no solution

    :Example:

    >>> v = lt_solve(D, z, q=2)
    """
    _check_prime(q)
    D = np.asarray(D) % q
    z = np.asarray(z) % q
    m, n = D.shape
    if q**m > 2**22:
        raise ValueError(f'too many rows {m}, q^m must be <= 2^22')

    # choose implementation
    backend = tools.get_backend()
    if backend == tools.BACKEND_RUST and q >= 256:
        warnings.warn(
            f'Rust backend of LT requires q < 256, got {q=}, '
            'falling back to Python',
            RuntimeWarning,
        )
        backend = tools.BACKEND_PYTHON
    if backend == tools.BACKEND_RUST:
        return rs.lt_solve(
            np.ascontiguousarray(D, dtype='uint8'),
            np.ascontiguousarray(z, dtype='uint8'),
            q,
        )
    elif backend == tools.BACKEND_PYTHON:
        return _lt_solve(D, z, q)
    else:
        raise NotImplementedError(f'unknown backend {backend}')


def _lt_solve(D: np.ndarray, z: np.ndarray, q: int) -> np.ndarray:
    """Python implementation of :func:`lt_solve`."""
    # columns and right-hand side as base-q integers
    indices, coefs = _coset_leader(
        _encode(D, q),
        int(_encode(z[:, None], q)[0]),
        q, len(D),
    )
    v = np.zeros(D.shape[1], dtype='uint8')
    v[indices] = coefs
    return v


def _coset_leader(
    columns: np.ndarray,
    s: int,
    q: int,
    p: int,
) -> tuple:
    """Finds the fewest columns, with coefficients, adding up to s.

    Columns and s are base-q integers of p digits.
    Set C_i holds the syndromes reachable with i columns, but not fewer.

    :return: tuple (indices, coefficients) of columns
    :rtype: tuple
    :raises ValueError: if s is not reachable
    """
    if s == 0:
        return np.zeros(0, dtype='int64'), np.zeros(0, dtype='int64')

    # workspace over all syndromes, lazily zeroed, so only touched pages cost
    size = q**p
    level = np.zeros(size, dtype='int8')  # i + 1 for syndrome in C_i, 0 if unseen
    parent = np.zeros(size, dtype='int64')  # syndrome with one column less
    column = np.zeros(size, dtype='int64')  # that column
    coef = np.zeros(size, dtype='int64')  # and its coefficient
    level[0] = 1  # C_0 holds the zero syndrome

    def path(v):
        """Columns and coefficients adding up to syndrome v."""
        while v != 0:
            yield column[v], coef[v]
            v = parent[v]

    def leader(*vs):
        """Columns and coefficients of syndromes vs, summed."""
        v = {}
        for u in vs:
            for j, a in path(u):
                v[j] = (v.get(j, 0) + a) % q
        indices = np.array([j for j in v if v[j]], dtype='int64')
        return indices, np.array([v[j] for j in indices], dtype='int64')

    # C_1, all multiples of the columns, first occurrence of each
    multiples = np.stack([_mul(a, columns, q, p) for a in range(1, q)], axis=1).ravel()
    values, first = np.unique(multiples, return_index=True)
    first, values = first[values != 0], values[values != 0]
    if len(values) == 0:
        raise ValueError('system has no solution')
    columns1, coefs1 = first // (q - 1), first % (q - 1) + 1
    level[values] = 2
    column[values] = columns1
    coef[values] = coefs1
    sets = [np.zeros(1, dtype='int64'), values]

    # single column
    if level[s] == 2:
        return leader(s)

    l = r = 1
    while True:
        # is there x in C_l, s.t. s - x in C_r
        x = sets[l]
        hit = np.flatnonzero(level[_add(s, _neg(x, q, p), q, p)] == r + 1)
        if len(hit):
            x = int(x[hit[0]])
            return leader(x, int(_add(s, _neg(x, q, p), q, p)))

        # next pair (l, r), creating C_r if needed
        if l < r:
            l += 1
            continue
        r += 1
        if r < len(sets):
            continue

        # C_r = C_{r-1} + C_1, without syndromes of fewer columns,
        # in chunks of the frontier to bound memory
        frontier = sets[-1]
        created = []
        chunk = max(1, 2**22 // len(values))
        for start in range(0, len(frontier), chunk):
            part = frontier[start:start+chunk]
            new = _add(part[:, None], values[None, :], q, p).ravel()
            unseen = level[new] == 0
            new, i = np.unique(new[unseen], return_index=True)
            level[new] = r + 1
            parent[new] = np.repeat(part, len(values))[unseen][i]
            column[new] = np.tile(columns1, len(part))[unseen][i]
            coef[new] = np.tile(coefs1, len(part))[unseen][i]
            created.append(new)
        new = np.concatenate(created)
        sets.append(new)
        if len(new) == 0:
            raise ValueError('system has no solution')
