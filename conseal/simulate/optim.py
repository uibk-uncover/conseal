"""

The equation is from

P. Bas, et al. "Break Our Steganographic System: The Ins and Outs of Organizing BOSS", IH 2011.

Originally described in the STC paper by
T. Filler, et al. "Minimizing Additive Distortion in Steganography Using Syndrome-Trellis Codes", TIFS 2011.


Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import enum
import numpy as np
from typing import Callable, Tuple
import warnings

from .. import tools
from . import optim_new
from ._defs import Sender


def get_p(
    lbda: float,
    *rhos: np.ndarray,
    add_zero: bool = True,
) -> np.ndarray:
    """Converts distortions into probabilities,
    using Boltzmann-Gibbs distribution

    For more details, see `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#embedding-simulation>`__.

    :param rhos: distortion of embedding choices, e.g. embedding +1 or embedding -1
    :type rhos: tuple
    :param lbda: parameter value
    :type lbda: float
    :param add_zero:
    :type add_zero: bool
    :param p_pm1: probability tensor for changes associated to rhos[0]
        of an arbitrary shape
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    warnings.warn('get_p is deprecated, use get_probability instead', DeprecationWarning)
    # denominator (forced left-associativity)
    denum = 1 if add_zero else 0
    for rho in rhos:
        denum += np.exp(-lbda * rho)
    #
    denum[denum == 0] = tools.EPS
    return np.exp(-lbda * rhos[0]) / denum


def average_payload(
    *,
    ps: Tuple[np.ndarray] = None,
    e: float = None,
    lbda: float = None,
    rhos: Tuple[np.ndarray] = None,
    q: int = None,
) -> Tuple[Tuple[np.ndarray], float]:
    """

    :param ps: Probability maps.
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param e: Embedding efficiency. If not provided, optimal coding is assumed.
    :type e: float
    :param lbda: Parameter of the Gibbs distribution, if rhos are given.
    :type lbda: float
    :param rhos: Cost maps. Can be provided instead of `ps`.
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return:
    :rtype: float

    :Example:

    >>> # TODO
    """
    # assert (
    #     (ps is not None and rhos is None or ps is None and rhos is not None)
    # ), 'one of ps or rhos must be given'
    # assert (
    #     lbda is not None or ps is not None
    # ), 'lbda can be specified only with rhos'
    #
    if ps is None:
        add_zero = True if q is None else len(rhos) == q-1
        ps = [
            get_p(lbda, rhos[i], *rhos[:i], *rhos[i+1:], add_zero=add_zero)
            for i in range(len(rhos))
        ]

    # Imperfect coding - given embedding efficiency
    if e is not None:
        H = np.sum(ps) * e
    # Perfect coding - upper bound efficiency
    elif q is None or len(ps) == q-1:  # no change is zero cost
        H = tools.entropy(*ps)
    else:
        H = tools._entropy(*ps)

    return ps, H


def d_average_payload(
    *,
    ps: Tuple[np.ndarray] = None,
    e: float = None,
    lbda: float = None,
    rhos: Tuple[np.ndarray] = None,
    q: int = None,
) -> float:
    # assert (
    #     (ps is not None and rhos is None or ps is None and rhos is not None)
    # ), 'one of ps or rhos must be given'
    # assert (
    #     lbda is not None or ps is not None
    # ), 'lbda can be specified only with rhos'
    #
    add_zero = True if q is None else len(rhos) == q-1
    if ps is None:
        ps = [
            get_p(lbda, rhos[i], *rhos[:i], *rhos[i+1:], add_zero=add_zero)
            for i in range(len(rhos))
        ]

    # Imperfect coding - given embedding efficiency
    if e is not None:
        raise NotImplementedError
    else:
        px = np.array(list(ps))
        if add_zero:
            px0 = 1-np.sum(px, axis=0)[None]
            px = np.concatenate([px, px0], axis=0)

        px[px <= 0] = 1  # avoid log(0)
        p = px[:-1]
        p0 = px[-1:]
        E_rho = np.sum(p * rhos, axis=0, keepdims=True)
        dp_dlambda = p * (E_rho - rhos)
        log2_term = np.log2(p0) - np.log2(p)
        dH = dp_dlambda * log2_term

        # px[px <= 0] = 1  # avoid log(0)
        # log2_px = np.log2(px)
        # log2_recip = 1.4426950408889634  # 1 / np.log2(2)
        # dH = (log2_px + log2_recip) * (log2_px - log2_px[-1:])
    return np.sum(dH)


def average_distortion(
    rhos: Tuple[np.ndarray],
    *,
    # e: float = None,
    lbda: float = None,
    ps: Tuple[np.ndarray] = None,
    # q: int = None,
) -> Tuple[Tuple[np.ndarray], float]:
    """

    :param ps: Probability maps.
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param e: Embedding efficiency. If not provided, optimal coding is assumed.
    :type e: float
    :param lbda: Parameter of the Gibbs distribution, if rhos are given.
    :type lbda: float
    :param rhos: Cost maps. Can be provided instead of `ps`.
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return:
    :rtype: float

    :Example:

    >>> # TODO
    """
    assert (
        (ps is not None and lbda is None or ps is None and lbda is not None)
    ), 'one of ps or lbda must be given'
    #
    if ps is None:
        # add_zero = True if q is None else len(rhos) == q-1
        add_zero = True
        ps = [
            get_p(lbda, rhos[i], *rhos[:i], *rhos[i+1:], add_zero=add_zero)
            for i in range(len(rhos))
        ]

    # average distortion
    D = np.sum([
        rhos[i] * ps[i]
        for i in range(len(rhos))
    ])
    return ps, D


def get_objective(
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    e: float = None,
    q: int = None,
) -> Callable:
    def _pls_objective(*args, **kw):
        return average_payload(*args, e=e, q=q, **kw)

    if sender == Sender.PAYLOAD_LIMITED_SENDER:
        # print('selected PLS objective')
        return average_payload if e is None else _pls_objective
    elif sender == Sender.DISTORTION_LIMITED_SENDER:
        # print('selected DLS objective')
        assert e is None, 'e not implemented for DLS'
        return average_distortion
    else:
        raise NotImplementedError(f'unknown sender {sender}')


def get_d_objective(
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    e: float = None,
    q: int = None,
) -> Callable:
    def _d_pls_objective(*args, **kw):
        return d_average_payload(*args, e=e, q=q, **kw)

    if sender == Sender.PAYLOAD_LIMITED_SENDER:
        # print('selected PLS objective')
        return d_average_payload if e is None else _d_pls_objective
    elif sender == Sender.DISTORTION_LIMITED_SENDER:
        raise NotImplementedError('DiLS not implemented for Newton')
    else:
        raise NotImplementedError(f'unknown sender {sender}')


def binary_search_lambda(
    rhos: Tuple[np.ndarray],
    m: int,
    n: int,
    objective: Callable = None,
    **kw
) -> float:
    """Implements binary search for lambda.

    The i-th element is embedded with a probability of
    p_i = 1/Z exp( -lambda D(X, y_i X_{~i}) ),
    where D is the distortion after modifying the i-th cover element, and Z is a normalization constant.
    This methods determines the lambda to communicate the message of the message length.

    We simulate a payload-limited sender that embeds a fixed average payload while minimizing the average distortion.
    Optimize for the lambda that minimizes the average distortion while transferring the message.

    :param rhos: Tuple of costs.
    :type rho_p1: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param m: Message length.
    :type m: int
    :param n: Cover size.
    :type n: int
    :return: Parameter lambda value.
    :rtype: float

    :Example:

    >>> # TODO
    """
    assert n > 0, "Expected cover size greater than 0"

    # Initialize lambda and m3 such that the loop is at least entered once
    # m3 is the total entropy
    l3 = 1000
    m3 = float(m + 1)

    # Initialize iteration counter
    iterations = 0

    # Find the largest l3, s.t., H(m3) <= m
    while m3 > m:

        # Increase l3
        l3 *= 2

        # Compute total entropy m3
        _, m3 = objective(lbda=l3, rhos=rhos)  # objective function

        iterations += 1

        # unbounded = search fails
        if iterations > 15:
            warnings.warn("unbounded distortion, search fails", RuntimeWarning)
            return l3

    # Initialize lower bound to zero
    l1 = 0
    # The lower bound for the message size is n
    m1 = float(n)
    alpha = float(m) / n  # embedding rate

    # Binary search for lambda
    # Relative payload must be within 1e-3 of the required relative payload
    while float(m1 - m3) / n > alpha / 1000 and iterations < 30:
        # Mid of the interval [l1, l3]
        lbda = l1 + (l3 - l1) / 2

        # Calculate entropy at the mid of the interval
        _, m2 = objective(lbda=lbda, rhos=rhos)  # objective function

        # binary search
        if m2 < m:
            # The average payload is too small for the message.
            # We need to decrease the upper bound
            l3 = lbda
            m3 = m2
        else:
            # The average payload exceeds the message length
            # We can increase the lower bound
            l1 = lbda
            m1 = m2

        # Proceed to the next iteration
        iterations = iterations + 1

    if iterations == 30:
        warnings.warn("optimization might not have converged", RuntimeWarning)

    return lbda


# class LambdaOptimizer(enum.Enum):
#     """Type of lambda optimizer."""

#     BINARY_SEARCH = enum.auto()
#     """Binary search."""
#     NEWTON = enum.auto()
#     """Newton method."""

#     def __call__(self, *args, **kw):
#         global binary_search_lambda, _optim_binary
#         if self == LambdaOptimizer.BINARY_SEARCH:
#             return binary_search_lambda(*args, **kw)
#         else:
#             raise NotImplementedError(f"No implementation for optimizer {self}")


