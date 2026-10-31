"""Objectives of the λ search, their derivatives and limits.

Based on the Gibbs distribution approach for minimal embedding distortion.

Author: Martin Benes, Ilyas Satik, Marco Cotrotzo
Affiliation: University of Innsbruck
"""

import functools
import numpy as np
from typing import Tuple, Callable

from ._common import Sender, get_probability, _finite_rhos, _variance


def distortion(
    rhos: np.ndarray | Tuple[np.ndarray],
    lbda: float = None,
    *,
    ps: np.ndarray | Tuple[np.ndarray] = None,
    q: int = None,
) -> float:
    """
    Compute the expected distortion E[D] under the Gibbs distribution for
    a binary embedding model:
        p_i = 1 / (1 + exp(lambda * rho_i))
        E[D] = sum_i (rho_i * p_i)

    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param lbda: inverse temperature λ
    :type lbda: float
    :return: value of the expected distortion
    :rtype: float
    """
    #
    rhos = np.array(rhos)
    # calculate probabilities
    add_zero = True if q is None else len(rhos) == q-1
    if ps is None:
        assert lbda is not None, 'either lambda or ps must be provided'
        ps = get_probability(lbda=lbda, rhos=rhos, add_zero=add_zero)
    # expected distortion
    Erho = np.sum(ps * _finite_rhos(rhos, ps))
    return float(Erho)


def d_distortion(
    rhos: np.ndarray,
    lbda: float = None,
    *,
    ps: np.ndarray | Tuple[np.ndarray] = None,
    q: int = None,
) -> float:
    """
    Compute the derivative of expected distortion with respect to λ:
        dE[D]/dλ = -sum_i Var_p(rho_i)

    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param lbda: inverse temperature λ
    :type lbda: float
    :return: value of the derivative of the expected distortion
    :rtype: float
    """
    #
    rhos = np.array(rhos)
    # calculate probabilities
    add_zero = True if q is None else len(rhos) == q-1
    if ps is None:
        assert lbda is not None, 'either lambda or ps must be provided'
        ps = get_probability(lbda=lbda, rhos=rhos, add_zero=add_zero)
    # dE[D]/dλ = -sum_i Var_p(ρ_i)
    return -float(np.sum(_variance(rhos, ps)))


def entropy(
    rhos: np.ndarray,
    lbda: float = None,
    *,
    ps: np.ndarray | Tuple[np.ndarray] = None,
    q: int = None,
    e: float = None,
) -> float:
    """Compute the entropy H(p) of probability p.
    Logit is provided for better numerical stability.

    H = -sum p log2 p + (1-p) log2 (1-p)
    = -1/ln2 * sum (p L - log(1+exp(L)))

    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param lbda: inverse temperature λ
    :type lbda: float
    :return: value of the derivative of the entropy
    :rtype: float
    """
    #
    rhos = np.array(rhos)
    # calculate probabilities
    add_zero = True if q is None else len(rhos) == q-1
    if ps is None:
        assert lbda is not None, 'either lambda or ps must be provided'
        # print(f'get probability with {lbda=} {rhos.shape=} {add_zero=}')
        ps = get_probability(lbda=lbda, rhos=rhos, add_zero=add_zero)

    # Imperfect coding - given embedding efficiency
    if e is not None:
        H = np.sum(ps) * e
    # Perfect coding - upper bound efficiency
    elif add_zero:
        ps = np.concatenate([
            1-np.sum(ps, axis=0, keepdims=True),
            ps,
        ], axis=0)
        ps = np.where(ps < 1e-10, 1.0, ps)
        H = -np.nansum(ps * np.log2(ps))
    else:
        ps = np.where(ps < 1e-10, 1.0, ps)
        H = -np.sum(ps * np.log2(ps))
    #     H = tools.entropy(*ps)
    # else:
    #     ps = np.where(ps < 1e-10, 1.0, ps)
    #     H = -np.sum(ps * np.log2(ps))
        # H = tools._entropy(*ps)

    return float(H)


def d_entropy(
    rhos: np.ndarray,
    lbda: float = None,
    *,
    ps: np.ndarray | Tuple[np.ndarray] = None,
    q: int = None,
    e: float = None,
) -> float:
    """Compute the derivative dH/dλ of the entropy H(p),
    where p is the probability map and λ is the inverse entropy.
    Logit is provided for better numerical stability.

    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param lbda: inverse temperature λ
    :type lbda: float
    :return: value of the derivative of the entropy
    :rtype: float
    """
    if e is not None:
        raise NotImplementedError('Newton method with imperfect coding not implemented')
    #
    rhos = np.array(rhos)
    # calculate probabilities
    add_zero = True if q is None else len(rhos) == q-1
    if ps is None:
        assert lbda is not None, 'either lambda or ps must be provided'
        ps = get_probability(lbda=lbda, rhos=rhos, add_zero=add_zero)
    # dH/dλ = -λ/ln2 * sum_i Var_p(ρ_i)
    return float(-lbda / np.log(2) * np.sum(_variance(rhos, ps)))


def _limit_states(
    rhos: np.ndarray | Tuple[np.ndarray],
    *,
    q: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """States of each element at the limits λ -> 0+ and λ -> inf.

    As λ -> 0+, all states with finite cost become equally likely.
    As λ -> inf, only the states with minimal cost remain, equally likely.

    :return: tuple (states, finite, rho_min, tied), where
        states are the costs including the no-change state if implicit,
        finite marks states reachable as λ -> 0+, and
        tied marks states remaining as λ -> inf
    :rtype: tuple
    """
    rhos = np.array(rhos, dtype=float)
    add_zero = True if q is None else len(rhos) == q-1
    if add_zero:
        rhos = np.concatenate([np.zeros_like(rhos[:1]), rhos])
    finite = np.isfinite(rhos)
    rho_min = np.min(np.where(finite, rhos, np.inf), axis=0)
    tied = rhos == rho_min
    return rhos, finite, rho_min, tied


def entropy_limits(
    rhos: np.ndarray | Tuple[np.ndarray],
    *,
    q: int = None,
    e: float = None,
) -> Tuple[float, float]:
    """Limits H(0+) and H(inf) of the entropy, in closed form.

    :param rhos: embedding costs
    :type rhos: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param q: number of states
    :type q: int
    :param e: embedding efficiency, optimal coding by default
    :type e: float
    :return: tuple (H(0+), H(inf))
    :rtype: tuple
    """
    states, finite, _, tied = _limit_states(rhos, q=q)
    k_0 = np.sum(finite, axis=0)
    k_inf = np.sum(tied, axis=0)
    # Imperfect coding - change rate times efficiency
    if e is not None:
        # states given by rhos, i.e., without the implicit no-change
        n_given = len(np.array(rhos))
        p_0 = np.sum(finite[-n_given:], axis=0) / k_0
        p_inf = np.sum(tied[-n_given:], axis=0) / k_inf
        return float(e * np.sum(p_0)), float(e * np.sum(p_inf))
    # Perfect coding - uniform over states
    return float(np.sum(np.log2(k_0))), float(np.sum(np.log2(k_inf)))


def distortion_limits(
    rhos: np.ndarray | Tuple[np.ndarray],
    *,
    q: int = None,
) -> Tuple[float, float]:
    """Limits E[D](0+) and E[D](inf) of the expected distortion, in closed form.

    :param rhos: embedding costs
    :type rhos: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param q: number of states
    :type q: int
    :return: tuple (E[D](0+), E[D](inf))
    :rtype: tuple
    """
    states, finite, rho_min, _ = _limit_states(rhos, q=q)
    rho_mean = np.sum(np.where(finite, states, 0.), axis=0) / np.sum(finite, axis=0)
    return float(np.sum(rho_mean)), float(np.sum(rho_min))


class Objective:
    """Objective f(λ) of the λ search, with its derivative and limits.

    The constructor dispatches the implementation by the sender,
    or wraps custom callables, if value is given.

    :param sender: type of sender
    :type sender: :class:`Sender`
    :param e: embedding efficiency, optimal coding by default
    :type e: float
    :param value: custom objective f(rhos, lbda, ps=, q=)
    :type value: callable
    :param derivative: custom derivative df/dλ, same signature as value
    :type derivative: callable
    :param limits: custom limits (f(0+), f(inf)), signature limits(rhos, q=)
    :type limits: callable

    :Example:

    >>> objective = Objective(cl.PLS)
    >>> H = objective(rhos, lbda)
    >>> dH = objective.derivative(rhos, lbda)
    >>> H_0, H_inf = objective.limits(rhos)
    """

    def __init__(
        self,
        sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
        *,
        e: float = None,
        value: Callable = None,
        derivative: Callable = None,
        limits: Callable = None,
    ):
        self.sender = sender
        self.e = e
        # custom objective
        if value is not None:
            self.sender = None
            self._value = value
            self._derivative = derivative
            self._limits = limits
        # payload-limited sender
        elif sender == Sender.PAYLOAD_LIMITED_SENDER:
            self._value = functools.partial(entropy, e=e)
            self._derivative = functools.partial(d_entropy, e=e)
            self._limits = functools.partial(entropy_limits, e=e)
        # distortion-limited sender
        elif sender == Sender.DISTORTION_LIMITED_SENDER:
            if e is not None:
                raise NotImplementedError('embedding efficiency not applicable to distortion')
            self._value = distortion
            self._derivative = d_distortion
            self._limits = distortion_limits
        else:
            raise NotImplementedError(f'unknown sender {sender}')

    @classmethod
    def of(
        cls,
        objective: 'Sender | Objective | Callable',
        *,
        e: float = None,
    ) -> 'Objective':
        """Coerces sender, objective or callable into an objective.

        :param objective: sender to dispatch by,
            objective to use as is, or
            callable to wrap as objective without derivative and limits
        :type objective: :class:`Sender`, :class:`Objective` or callable
        :param e: embedding efficiency, only with sender
        :type e: float
        :return: objective
        :rtype: :class:`Objective`
        """
        if isinstance(objective, Sender):
            return cls(objective, e=e)
        if e is not None:
            raise ValueError('embedding efficiency e can be given only with sender')
        if isinstance(objective, cls):
            return objective
        if callable(objective):
            return cls(value=objective)
        raise TypeError(f'expected sender, objective or callable, got {objective!r}')

    def __call__(self, *args, **kw) -> float:
        """Value of the objective f(λ)."""
        return self._value(*args, **kw)

    def derivative(self, *args, **kw) -> float:
        """Derivative of the objective df/dλ."""
        if self._derivative is None:
            raise NotImplementedError('objective has no derivative')
        return self._derivative(*args, **kw)

    def limits(self, rhos: np.ndarray | Tuple[np.ndarray], *, q: int = None) -> Tuple[float, float]:
        """Limits (f(0+), f(inf)) of the objective, None if unknown."""
        if self._limits is None:
            return None
        return self._limits(rhos, q=q)

    def __repr__(self) -> str:
        if self.sender is None:
            return f'Objective({self._value})'
        return f'Objective({self.sender.name}, e={self.e})'
