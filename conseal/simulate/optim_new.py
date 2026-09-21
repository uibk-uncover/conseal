"""
Implementation of λ estimation methods for the steganography simulator.

Based on the Gibbs distribution approach for minimal embedding distortion.
Provides several root-finding strategies to solve E[D](λ) = target_distortion.

Author: Martin Benes, Ilyas Satik, Marco Cotrotzo
Affiliation: University of Innsbruck
"""

import enum
import logging
import numpy as np
from typing import Tuple, Callable
import warnings

from ._defs import Sender
from .. import tools



def get_probability(
    rhos: np.ndarray | Tuple[np.ndarray],
    lbda: float,
    *,
    add_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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
    rhos = np.array(rhos)
    exponent = -lbda * rhos

    # Log-Sum-Exp stabilization
    exponent_max = np.max(exponent, axis=0, keepdims=True)
    if add_zero:
        exponent_max = np.maximum(0.0, exponent_max)
    exponent_stable = exponent - exponent_max
    gibbs = np.exp(exponent_stable)

    #
    denum = np.exp(-exponent_max) if add_zero else 0.0
    return gibbs / np.clip(denum + np.nansum(gibbs, axis=0, keepdims=True), 1e-15, None)


# === OBJECTIVES ===

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
    Erho = np.sum(ps * rhos)
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
        dE[D]/dλ = -sum_i (rho_i^2 * p_i * (1 - p_i))

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
    # expected distortion
    return np.sum(-np.sum(ps * rhos**2, axis=0) + np.sum(ps * rhos, axis=0)**2)
    # #
    # E_rho = np.sum(ps * rhos, axis=0)
    # active_variance = np.sum(ps * (rhos - E_rho) ** 2, axis=0)
    # if add_zero:
    #     p0 = 1.0 - np.sum(ps, axis=0)
    #     zero_state_variance = p0 * (E_rho ** 2)
    #     variance = active_variance + zero_state_variance
    # else:
    #     variance = active_variance
    # return -float(np.sum(variance))


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
    # entropy
    Erho = np.sum(rhos * ps, axis=0, keepdims=True)
    ps = np.where(ps < 1e-10, 1.0, ps)
    return np.sum(np.sum(ps * np.log2(ps) * (rhos - Erho), axis=0))


def get_objective(
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    *,
    e: float = None,
) -> Callable:
    if sender == Sender.PAYLOAD_LIMITED_SENDER:
        if e is not None:
            return lambda *args, **kw: entropy(*args, e=e, **kw)
        else:
            return entropy
    elif sender == Sender.DISTORTION_LIMITED_SENDER:
        if e is not None:
            return lambda *args, **kw: distortion(*args, e=e, **kw)
        else:
            return distortion
    else:
        raise NotImplementedError(f'unknown sender {sender}')


def get_d_objective(
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    *,
    e: float = None,
) -> Callable:
    # assert e is None, 'e not implemented'
    if sender == Sender.PAYLOAD_LIMITED_SENDER:
        if e is not None:
            return lambda *args, **kw: d_entropy(*args, e=e, **kw)
        else:
            return d_entropy
    elif sender == Sender.DISTORTION_LIMITED_SENDER:
        if e is not None:
            return lambda *args, **kw: d_distortion(*args, e=e, **kw)
        else:
            return d_distortion
    else:
        raise NotImplementedError(f'unknown sender {sender}')

# === SOLVERS ===

def binary_search(
    target: float,
    rhos: np.ndarray,
    *,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    objective: Callable = None,
    d_objective: Callable = None,
    lbda1: Tuple[float] = None,
    tol: float = 1,
    n: int = None,  # legacy
    max_iter: int = None,
    q: int = None,
    e: float = None,
) -> float:
    """Binary search to solve J(x) = target up to tolerance.

    :param target: desired value
    :type target: float
    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param tol: search tolerance
    :type tol: float
    :return: estimated λ value
    :rtype: float
    """
    if objective is None:
        objective = get_objective(sender=sender, e=e)
    if lbda1 is None:
        lbda1 = (0, 1e4)
    # if n is None:
    #     n = rhos[0].size
    if q is None:
        q = len(rhos) + 1
    if max_iter is None:
        max_iter = 30
    #
    low, high = lbda1

    logging.debug(f'BSearch started, {target=} {low=} {high=} {tol=} {objective=} {d_objective=}')
    for it in range(max_iter):

        lbda = .5 * (low + high)
        f_val = objective(rhos=rhos, lbda=lbda, q=q)
        logging.debug(f'  {it=} | {lbda=:.03g} {f_val=:.03f} | {target=}')
        if abs(f_val - target) < tol:
            break

        if f_val > target:
            low = lbda
        else:
            high = lbda
    else:
        warnings.warn("optimization might not have converged", RuntimeWarning)
    lbda = .5 * (low + high)
    return lbda  # get_probability(rhos=rhos, lbda=lbda)[0], lbda


def newton(
    target: float,
    rhos: np.ndarray,
    *,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    objective: Callable = None,
    d_objective: Callable = None,
    lbda1: Tuple[float],
    tol: float = 1,
    n: int = None,  # legacy
    # add_zero: bool = True,
    max_iter: int = 30,
    q: int = None,
    e: float = None,
) -> float:
    """Newton-Raphson method to find λ satisfying E[D](λ) = target.

    :param target: Desired value
    :type target: float
    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param objective:
    :type objective:
    :param d_objective:
    :type d_objective:
    :param initial: Initial guess for λ
    :type initial: float
    :param tol: tolerance for convergence
    :type tol: float
    :param max_iter: Maximum iterations
    :type max_iter: int
    :return: Estimated λ value
    """
    if objective is None:
        objective = get_objective(sender=sender, e=e)
    if d_objective is None:
        d_objective = get_d_objective(sender=sender, e=e)
    if n is None:
        n = rhos[0].size
    if q is None:
        q = len(rhos) + 1
    #
    rhos = np.array(rhos)
    lbda = lbda1[0]
    add_zero = True if q is None else len(rhos) == q-1
    #
    logging.debug(f'Newton started, {target=} {lbda=} {tol=} {objective=} {d_objective=}')
    for it in range(max_iter):

        ps = get_probability(rhos=rhos, lbda=lbda, add_zero=add_zero)

        f_val = objective(rhos=rhos, lbda=lbda, ps=ps, q=q)
        if abs(f_val - target) < tol:
            logging.debug(f'  {it=} | {lbda=:.03g} {f_val=:.03f} | {target=}')
            break

        df_val = d_objective(rhos=rhos, lbda=lbda, ps=ps, q=q)
        logging.debug(f'  {it=} | {lbda=:.03g} {f_val=:.03f} {df_val=:.03f} | {target=}')
        if np.abs(df_val) < 1e-10:
            warnings.warn('Newton search for lambda diverged')
            break
        lbda = lbda - (f_val - target) / df_val

    else:
        warnings.warn("Newton search for lambda did not converge", RuntimeWarning)

    return lbda
    # return get_probability(rhos=rhos, lbda=lbda)[0], lbda


def polynomial_proxy(
    target: float,
    rhos: np.ndarray,
    *,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    objective: Callable = None,
    d_objective: Callable = None,
    lbda1: Tuple[float] = None,
    tol: float = 1,
    n: int = None,
    max_iter: int = 15,
    q: int = None,
    e: float = None,
    deg: int = 2,
) -> float:
    """Fits a polynomial for a quick estimate.
    Accuracy crucially depends on model fit.

    Specifically, the function fits a polynomial
        y(λ) ≈ a_deg λ^deg + ... + a_1 λ^1 + a_0
    to sample points (λ, E[D](λ)), and then solves for λ where y(λ) = target.

    :param target: desired expected distortion
    :type target: float
    :param deg: degree of the polynomial to fit
    :type deg: int
    :param lbdas: λ sample points for polynomial fit
    :type lbdas: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: Estimated λ value or NaN if no valid real root
    """
    if objective is None:
        objective = get_objective(sender=sender, e=e)
    if lbda1 is None:
        lbda1 = (5, 30)
    if n is None:
        n = rhos[0].size
    if q is None:
        q = len(rhos) - 1
    #
    xs = np.array(lbda1, dtype=np.float64)
    ys = np.array([
        objective(lbda=lbda, rhos=rhos)
        for lbda in lbda1
    ], dtype=np.float64)
    coefs = np.polyfit(xs, ys, deg)
    coefs[-1] -= target
    roots = np.roots(coefs)
    real_roots = [r.real for r in roots if np.isreal(r) and (r.real > 0)]
    if real_roots:
        lbda = min(real_roots)
    else:
        warnings.warn("optimization might not have converged", RuntimeWarning)
        # print([np.abs(y - target) for y in ys], np.argmin([np.abs(ys - target) for lbda in lbda0]))
        lbda = xs[np.argmin([np.abs(ys - target) for lbda in lbda1])]
        # print(lbda)
    return get_probability(rhos=rhos, lbda=lbda)[0], lbda


def taylor_inverse(
    target: float,
    rhos: np.ndarray,
    *,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    objective: Callable = None,
    d_objective: Callable = None,
    lbda1: float = None,
    tol: float = 1,
    # n: int = None,
    max_iter: int = 15,
    q: int = None,
    e: float = None
) -> float:
    """
    Single-step linear Taylor approximation: expand E[D](λ) around λ0, then invert.

    λ ≈ λ0 + (target - E[D](λ0)) / E'[D](λ0)

    :param target: desired expected distortion
    :type target: float
    :param lbda0: expansion point for Taylor (default 1.0)
    :type lbda0: float
    :return: estimated λ value (no iteration)
    :rtype: float
    """
    if objective is None:
        objective = get_objective(sender=sender, e=e)
    if d_objective is None:
        d_objective = get_d_objective(sender=sender, e=e)
    # if n is None:
    #     n = rhos[0].size
    if q is None:
        q = len(rhos) - 1
    if lbda1 is None:
        lbda1 = (5.,)
    # assess f(lbda) and f'(lbda)
    lbda = lbda1[0]
    f_val = objective(lbda=lbda, rhos=rhos)
    df_val = d_objective(lbda=lbda, rhos=rhos)
    # calculate lambda from Taylor
    lbda = lbda + (target - f_val) / df_val
    #
    return get_probability(rhos=rhos, lbda=lbda)[0], lbda


def expsearch_best(
    target: float,
    rhos: Tuple[np.ndarray],
    *,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    objective: Callable = None,
    d_objective: Callable = None,
    lbda0: Tuple[int] = None,
    max_iter: int = None,
    q: int = None,
    e: float = None,
) -> Tuple[Tuple[int], int]:

    if objective is None:
        objective = get_objective(sender=sender, e=e)
    if lbda0 is None:
        lbda0 = (-5, 15)
    #
    l0 = 0.0  # initial lambda
    d0 = np.inf  # initial distance
    #
    logging.debug(f'expsearch_best started, {target=} {lbda0=} {objective=}')
    for i, e in enumerate(range(*lbda0)):
        #
        lbda = 10**e
        f_val = objective(rhos=rhos, lbda=lbda, q=q)
        logging.debug(f'  {e=} | {lbda=:.03g} {f_val=:.03f} | {target=}')

        d = np.abs(f_val - target)
        if d < d0:
            d0, l0 = d, lbda
        elif i > 1:
            break  # dip passed
    else:
        warnings.warn("Exponential search for lambda did not converge", RuntimeWarning)
    #
    return (l0,), i


def expsearch_bound(
    target: float,
    rhos: Tuple[np.ndarray],
    *,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    objective: Callable = None,
    d_objective: Callable = None,
    lbda0: Tuple[int] = None,
    max_iter: int = None,
    q: int = None,
    e: float = None,
) -> Tuple[Tuple[int], int]:
    rhos = np.array(rhos)
    if objective is None:
        objective = get_objective(sender=sender, e=e)
    if lbda0 is None:
        lbda0 = (-5, 15)
    #
    logging.debug(f'expsearch_best started, {target=} {lbda0=} {objective=}')
    for i, e in enumerate(range(*lbda0)):
        #
        lbda1b = 10**e
        f_val = objective(rhos=rhos, lbda=lbda1b, q=q)
        logging.debug(f'  {e=} | {lbda1b=:.03g} {f_val=:.03f} | {target=}')

        if f_val < target:  # dip passed
            lbda1a = lbda1b/10 if i > 0 else 0  # TODO: negative lambda
            break
    else:
        warnings.warn("Exponential search for lambda did not converge", RuntimeWarning)
        lbda1a, lbda1b = lbda1b, np.finfo(rhos.dtype).max
    #
    return (lbda1a, lbda1b), i


class LambdaSolver(enum.Enum):
    """Type of lambda optimizer."""

    SOLVER_BSEARCH_DDE = enum.auto()
    """Binary search, compatible with DDE's Matlab."""
    SOLVER_BSEARCH = enum.auto()
    """Binary search."""
    SOLVER_NEWTON = enum.auto()
    """Newton method."""
    SOLVER_POLYPROXY = enum.auto()
    """Polynomial proxy method."""
    # TAYLOR_INVERSE = enum.auto()
    # """Taylor inverse method."""

    @property
    def solve(self):
        solvers = {
            LambdaSolver.SOLVER_NEWTON: newton,
            LambdaSolver.SOLVER_BSEARCH: binary_search,
            # SOLVER_BSEARCH_DDE: binary_search_dde,
            LambdaSolver.SOLVER_POLYPROXY: polynomial_proxy,
        }
        return solvers[self]

    @property
    def expsearch(self):
        expsearch = {
            LambdaSolver.SOLVER_NEWTON: expsearch_best,
            LambdaSolver.SOLVER_BSEARCH: expsearch_bound,
            # SOLVER_BSEARCH_DDE: binary_search_dde,
            # LambdaSolver.SOLVER_POLYPROXY: polynomial_proxy,
        }
        return expsearch[self]


def search(
    target: float,
    rhos: Tuple[np.ndarray],
    *,
    solver: LambdaSolver = LambdaSolver.SOLVER_NEWTON,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    objective: Callable = None,
    d_objective: Callable = None,
    lbda0: float | Tuple[int] = None,
    tol: float = 1,  # bits
    max_iter: int = 30,
    n: int = None,  # legacy
    q: int = None,
    e: float = None,
) -> float:
    """"""
    if objective is None:
        objective = get_objective(sender=sender, e=e)
    if d_objective is None:
        d_objective = get_d_objective(sender=sender, e=e)
    # if n is None:
    #     n = rhos[0].size
    if q is None:
        q = len(rhos) + 1
    #
    lbda1, _ = solver.expsearch(
        target=target,
        rhos=rhos,
        objective=objective,
        d_objective=d_objective,
        lbda0=lbda0,
        q=q,
    )
    #
    lbda = solver.solve(
        target=target,
        rhos=rhos,
        objective=objective,
        d_objective=d_objective,
        lbda1=lbda1,
        tol=tol,
        max_iter=max_iter,
        q=q,
    )
    return lbda

# ===




def exponential_search_dde(
    target: float,
    rhos: Tuple[np.ndarray],
    *,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    objective: Callable = None,
    d_objective: Callable = None,
    lbda0: Tuple[float] = (1000,),
    tol: float = None,
    n: int = None,
    max_iter: int = 15,
    q: int = None,
) -> Tuple[float, float]:
    assert n > 0, "Expected cover size greater than 0"
    # if max_iter is None:
    #     max_iter = 15
    # if lbda0 is None:
    #     lbda0 = (1000,)
        # lbda0 = (n,)
    # print(f'estimate_lambda_search: {max_iter=} {lbda0=} {target=}')

    # Initialize lambda and m3 such that the loop is at least entered once
    # m3 is the total entropy
    l3 = lbda0[0]
    m3 = float(target + 1)

    # Initialize iteration counter
    iterations = 0

    # Find the largest l3, s.t., H(m3) <= m
    while m3 > target:

        # Increase l3
        l3 *= 2

        # Compute total entropy m3
        m3 = objective(lbda=l3, rhos=rhos)  # objective function
        # print(f' - {iterations=}: expand {l3} {m3}')

        iterations += 1

        # unbounded = search fails
        if iterations > max_iter:
            warnings.warn("unbounded distortion, search fails", RuntimeWarning)
            break

    return (l3, m3), iterations


def search_dde(
    target: float,
    rhos: Tuple[np.ndarray],
    objective: Callable,
    d_objective: Callable,
    *,
    lbda0: Tuple[float] = None,
    tol: float = None,
    n: int = None,
    max_iter: int = None,
    # q: int = None,
    # m: int,
    # n: int,
    # objective: Callable = None,
    # **kw
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
    m = target
    if tol is None:
        tol = 1e-3
    if max_iter is None:
        max_iter = 30

    (l3, m3), iterations = boundary_search(
        target=target,
        rhos=rhos,
        objective=objective,
        d_objective=d_objective,
        lbda0=lbda0,
        tol=tol,
        n=n,
        # max_iter=15 if max_iter is None else max_iter//2,
    )

    # Initialize lower bound to zero
    l1 = 0
    # The lower bound for the message size is n
    m1 = float(n)
    alpha = float(m) / n  # embedding rate
    lbda = l1 + (l3 - l1) / 2

    # Binary search for lambda
    # Relative payload must be within 1e-3 of the required relative payload
    while float(m1 - m3) / n > alpha * tol and iterations < max_iter:
        # Mid of the interval [l1, l3]
        lbda = l1 + (l3 - l1) / 2

        # Calculate entropy at the mid of the interval
        m2 = objective(lbda=lbda, rhos=rhos)  # objective function

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

    if iterations == max_iter:
        warnings.warn("optimization might not have converged", RuntimeWarning)

    return get_probability(rhos=rhos, lbda=lbda)[0], lbda
    # return lbda


class LambdaOptimizer(enum.Enum):
    """Type of lambda optimizer."""

    BINARY_SEARCH_DDE = enum.auto()
    """Binary search, compatible with DDE's Matlab."""
    BINARY_SEARCH = enum.auto()
    """Binary search."""
    NEWTON = enum.auto()
    """Newton method."""
    POLYNOMIAL_PROXY = enum.auto()
    """Polynomial proxy method."""
    TAYLOR_INVERSE = enum.auto()
    """Taylor inverse method."""

    def __call__(self, *args, **kw):
        if self == LambdaOptimizer.BINARY_SEARCH_DDE:
            # print('using binary search as optimizer')
            return binary_search_dde(*args, **kw)
        elif self == LambdaOptimizer.BINARY_SEARCH:
            # print('using binary search as optimizer')
            return binary_search(*args, **kw)
        elif self == LambdaOptimizer.NEWTON:
            # print('using newton as optimizer')
            # default initial value obtained via initial lambda search
            if 'lbda0' not in kw:
                (lbda0, _), _ = estimate_lambda_search(*args, **kw)
                kw['lbda0'] = (lbda0 * .75,)
            # apply newton
            return newton(*args, **kw)
        elif self == LambdaOptimizer.POLYNOMIAL_PROXY:
            # print('using polynomial proxy as optimizer')
            return polynomial_proxy(*args, **kw)
        elif self == LambdaOptimizer.TAYLOR_INVERSE:
            # print('using taylor inverse as optimizer')
            return taylor_inverse(*args, **kw)

        else:
            raise NotImplementedError(f"No implementation for optimizer {self}")


# def compare_methods() -> list:
#     """
#     Compare four λ‐estimation methods on a grid of target distortions:
#       1) binary_search_lambda
#       2) newton_lambda
#       3) polynomial_proxy_lambda
#       4) taylor_inverse_lambda

#     Each method is timed (in ms) and its absolute error recorded against a
#     high‐precision “true” λ given by binary_search_lambda(..., tol=1e-9).
#     Returns a list of dicts with keys:
#       'target_distortion', 'true_lambda',
#       'binary_lambda', 'binary_time_ms', 'binary_error',
#       'newton_lambda', 'newton_time_ms', 'newton_error',
#       'poly_lambda', 'poly_time_ms', 'poly_error',
#       'taylor_lambda','taylor_time_ms','taylor_error'
#     """
#     # Create five target‐distortions between E[D](0.1) and E[D](5.0)
#     targets = np.linspace(
#         expected_distortion(0.1),
#         expected_distortion(5.0),
#         num=5
#     )
#     # Use a tighter binary search as “true” λ
#     true_lambdas = [binary_search_lambda(t, tol=1e-9) for t in targets]

#     records = []
#     for t, true_l in zip(targets, true_lambdas):
#         rec = {'target_distortion': t, 'true_lambda': true_l}

#         # 1) Binary Search (default tol=1e-6)
#         t0 = time.perf_counter()
#         bs = binary_search_lambda(t, tol=1e-6)
#         rec['binary_lambda'] = bs
#         rec['binary_time_ms'] = (time.perf_counter() - t0) * 1e3
#         rec['binary_error'] = abs(bs - true_l)

#         # 2) Newton’s Method
#         t0 = time.perf_counter()
#         nt = newton_lambda(t, initial=1.0)
#         rec['newton_lambda'] = nt
#         rec['newton_time_ms'] = (time.perf_counter() - t0) * 1e3
#         rec['newton_error'] = abs(nt - true_l)

#         # 3) Polynomial Proxy
#         t0 = time.perf_counter()
#         pp = polynomial_proxy_lambda(t)
#         rec['poly_lambda'] = pp
#         rec['poly_time_ms'] = (time.perf_counter() - t0) * 1e3
#         rec['poly_error'] = (abs(pp - true_l) if not math.isnan(pp) else float('nan'))

#         # 4) Taylor Inverse (linear around lambda0=1.0)
#         t0 = time.perf_counter()
#         tv = taylor_inverse_lambda(t, lambda0=1.0)
#         rec['taylor_lambda'] = tv
#         rec['taylor_time_ms'] = (time.perf_counter() - t0) * 1e3
#         rec['taylor_error'] = abs(tv - true_l)

#         records.append(rec)

#     return records


# if __name__ == "__main__":
#     # If run directly, print out comparison results in key=value format
#     results = compare_methods()
#     for rec in results:
#         line = ", ".join(
#             f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
#             for k, v in rec.items()
#         )
#         print(line)
