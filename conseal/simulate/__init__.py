"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import typing

from ._defs import get_p
from . import _ternary
from ._ternary import ternary


def simulate(
    rho: typing.Union[np.ndarray, typing.Tuple[np.ndarray]],
    alpha: float,
    n: int,
    seed: int = None,
    **kw,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray]]:
    """

    :param rho:
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param alpha:
    :type alpha: float
    :param n:
    :type n: int
    :param seed:
    :type seed: int
    :return:
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_

    :Example:

    >>> # TODO
    """
    # multiple rhos given
    if not isinstance(rho, np.ndarray):
        q = len(rho) + 1
    # single rho given
    else:
        q = 2

    # ternary
    if q == 3:
        return _ternary.ternary(
            rho[0], rho[1],
            alpha=alpha,
            n=n,
            seed=seed,
            **kw,
        )
    # other
    else:
        raise NotImplementedError(f'{q=} not implemented')


def average_payload(
    lbda: float,
    rho_pm1: np.ndarray = None,
    rho_p1: np.ndarray = None,
    rho_m1: np.ndarray = None,
    p_pm1: np.ndarray = None,
    p_p1: np.ndarray = None,
    p_m1: np.ndarray = None,
    q: int = 3,
) -> float:
    """

    :param lbda:
    :type lbda: float
    :param rho_pm1:
    :type rho_pm1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param rho_p1:
    :type rho_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param rho_m1:
    :type rho_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param p_pm1:
    :type p_pm1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param p_p1:
    :type p_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param p_m1:
    :type p_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
    :param q:
    :type q: int
    :return: 2-tuple of (p_p1, p_m1), alpha_hat, where
        p_p1 is the probability of +1 change,
        p_m1 is the probability of -1 change, and
        alpha_hat is the payload embedded.
    :rtype: tuple

    :Example:

    >>> # TODO
    """
    if q == 3:
        return _ternary.average_payload(
            lbda=lbda,
            rho_p1=rho_p1,
            rho_m1=rho_m1,
            p_p1=p_p1,
            p_m1=p_m1,
        )
    else:
        raise NotImplementedError(f'not implemented {q}ary probability')


__all__ = [
    '_defs',
    '_ternary',
    'ternary',
    'get_p',
    'simulate',
    'average_payload',
]
