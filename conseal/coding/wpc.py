"""Implementation of wet-paper codes.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np


def soliton(
    m: int = 100,
    *,
    robust: bool = True,
    c: float = .1,
    delta: float = .5,
) -> np.ndarray:
    """Calculates probabilities of Soliton distribution.

    Taken from Jessica's book, Eqs. 9.5 and 9.6.

    :param m: number of rows/message bits
    :type m: int
    :param robust: Set False to use ideal Soliton. By default True (robust).
    :type robust: bool
    :param c: constant parameter, 0.1 by default
    :type c: float
    :param delta: failure probability, 0.5 by default
    :type delta: float
    :return: probabilities corresponding to 1:m
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> i = np.arange(1, 101)
    >>> p_i = soliton(len(i))
    """
    # Soliton distribution
    i = np.arange(1, m+1)
    nu = np.concatenate([[1/m], 1/(i[1:]*i[:-1])])

    # robust Soliton distribution
    if robust:
        T = c * np.log(m / delta) * np.sqrt(m)
        mT = int(np.floor(m / T))
        # print(m, mT)
        assert m > mT
        tau = np.concatenate([
            T / (i[:mT-1] * m),
            [T * np.log(T / delta) / m],
            np.zeros(m - mT),
        ])
        p = nu + tau
    else:
        p = nu

    #
    return p / np.sum(p)


def generate_H(
    m: int,
    n: int = None,
    *,
    c: float = .1,
    delta: float = .5,
    seed: int = None,
) -> np.ndarray:
    """Generates parity-check matrix according to robust soliton distribution.

    :param m: number of rows/message bits
    :type m: int
    :param n: number of columns/cover elements
    :type n: int
    :param c: constant parameter, 0.1 by default
    :type c: float
    :param delta: failure probability, 0.5 by default
    :type delta: float
    :return: generated parity-check matrix
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> H = generate_H(50, delta=.5, c=.1, seed=12345)
    """
    if n is None:
        n = 2**m-1
    # Get the robust Soliton distribution
    rsd = soliton(m, c=c, delta=delta, robust=True)

    # Sample column weights w[1], ..., w[n] from the RSD
    rng = np.random.default_rng(seed)
    column_weights = rng.choice(m, size=n, p=rsd) + 1

    # Generate columns of H
    H = np.zeros((m, n), dtype=int)
    for j in range(n):
        # Create a column with column_weights[j] ones
        ones_positions = rng.choice(m, size=column_weights[j], replace=False)
        H[ones_positions, j] = 1

    return H
