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

    :param rho: either
        a distortion tensor for +-1 change, or
        a tuple with tensors for +1 and -1 change
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
    :type alpha: float
    :param n: Cover size.
    :type n: int
    :param seed: random seed for embedding simulator
    :type seed: int
    :return:
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y += cl.simulate.ternary(
    ...     rho_p1=rho_p1,  # distortion of +1
    ...     rho_m1=rho_m1,  # distortion of -1
    ...     alpha=0.4,  # alpha
    ...     n=im_dct.Y.size,  # cover size
    ...     seed=12345)  # seed
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

    You must provide one of following:
    - rho_pm1;
    - rho_p1 and rho_m1;
    - p_pm1; or
    - p_p1 and p_m1.

    :param lbda:
    :type lbda: float
    :param rho_pm1: distortion tensor for +-1 changes
        of an arbitrary shape
    :type rho_pm1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_p1: distortion tensor for +1 changes
        of an arbitrary shape
    :type rho_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_m1: distortion tensor for -1 changes
        of an arbitrary shape
    :type rho_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param p_pm1: probability tensor for changes
        of an arbitrary shape
    :type p_pm1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param p_p1: probability tensor for +1 changes
        of an arbitrary shape
    :type p_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param p_m1: probability tensor for -1 changes
        of an arbitrary shape
    :type p_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param q: q-arity of the code,
        3 by default
    :type q: int
    :return: 2-tuple of (p_p1, p_m1), alpha_hat, where
        p_p1 is the probability tensor of +1 change,
        p_m1 is the probability tensor of -1 change, and
        alpha_hat is the payload embedded.
    :rtype: tuple

    :Example:

    >>> (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
    ...     rho_p1=rho_p1,  # distortion of +1
    ...     rho_m1=rho_m1,  # distortion of -1
    ...     alpha=0.4,  # alpha
    ...     n=im_dct.Y.size)  # cover size
    >>> alpha_hat = cl.simulate._ternary.average_payload(
    ...     lbda=lbda,  # lambda (optimized)
    ...     rho_p1=rho_p1,  # distortion of +1
    ...     rho_m1=rho_m1)  # distortion of -1
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
