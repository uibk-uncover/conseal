"""Solvers of the λ search, f(λ) = target.

Author: Martin Benes, Ilyas Satik, Marco Cotrotzo
Affiliation: University of Innsbruck
"""

import enum
import logging
import numpy as np
from typing import Tuple, Callable
import warnings

from ._common import Sender, get_probability
from ._objective import Objective
from ._solver_brent import brent


def binary_search(
    target: float,
    rhos: np.ndarray,
    *,
    objective: Sender | Objective | Callable = Sender.PAYLOAD_LIMITED_SENDER,
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
    objective = Objective.of(objective, e=e)
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

    logging.debug(f'BSearch started, {target=} {low=} {high=} {tol=} {objective=}')
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


def binary_search_dde(
    target: float,
    rhos: np.ndarray,
    *,
    objective: Sender | Objective | Callable = Sender.PAYLOAD_LIMITED_SENDER,
    lbda1: Tuple[float] = None,
    tol: float = 1e-3,
    n: int = None,
    max_iter: int = 30,
    q: int = None,
    e: float = None,
    alpha_max: float = None,
) -> float:
    """Binary search for λ, replicating calc_lambda of DDE's Matlab.

    Doubles λ from lbda1 until f(λ) <= target, then bisects [0, λ]
    until the bracket's relative payload is within alpha * tol.
    Expansion and bisection share the iteration budget.

    :param target: desired value, message length m for PLS
    :type target: float
    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param lbda1: initial λ to double, 1000 in DDE
    :type lbda1: tuple
    :param tol: relative tolerance on embedding rate, 1/1000 in DDE
    :type tol: float
    :param n: cover size
    :type n: int
    :param alpha_max: maximal embedding rate,
        max(1, alpha) for binary and max(1, alpha+.1) for ternary by default
    :type alpha_max: float
    :return: estimated λ value
    :rtype: float
    """
    objective = Objective.of(objective, e=e)
    if lbda1 is None:
        lbda1 = (1000,)
    if n is None:
        n = rhos[0].size
    if q is None:
        q = len(rhos) + 1
    alpha = float(target) / n  # embedding rate
    if alpha_max is None:
        alpha_max = max(1., alpha if q == 2 else alpha + .1)
    #
    l3 = lbda1[0]
    m3 = float(target + 1)
    iterations = 0

    # find the largest l3, s.t., f(l3) <= target
    logging.debug(f'BSearch DDE started, {target=} {l3=} {tol=} {alpha_max=}')
    while m3 > target:
        l3 *= 2
        m3 = objective(rhos=rhos, lbda=l3, q=q)
        iterations += 1
        logging.debug(f'  expand {iterations=} | {l3=:.03g} {m3=:.03f}')
        # unbounded = search fails
        if iterations > 15:
            warnings.warn("unbounded distortion, search fails", RuntimeWarning)
            return l3

    # bisect [l1, l3]
    l1 = 0
    m1 = float(n * alpha_max)
    lbda = l1
    while float(m1 - m3) / n > alpha * tol and iterations < int(max_iter * alpha_max):
        lbda = l1 + (l3 - l1) / 2
        m2 = objective(rhos=rhos, lbda=lbda, q=q)
        logging.debug(f'  {iterations=} | {lbda=:.03g} {m2=:.03f} | {target=}')
        if m2 < target:
            l3, m3 = lbda, m2
        else:
            l1, m1 = lbda, m2
        iterations += 1

    if iterations == max_iter:
        warnings.warn("optimization might not have converged", RuntimeWarning)

    return lbda


def newton(
    target: float,
    rhos: np.ndarray,
    *,
    objective: Sender | Objective | Callable = Sender.PAYLOAD_LIMITED_SENDER,
    lbda1: Tuple[float] = None,
    tol: float = 1,
    n: int = None,  # legacy
    max_iter: int = 30,
    q: int = None,
    e: float = None,
    max_step: float = 2 * np.log(10),
) -> float:
    """Safeguarded Newton-Raphson method to find λ satisfying f(λ) = target.

    Solves log f(λ) = log target in u = log λ, which keeps λ > 0
    and makes the iterations invariant to the scale of the costs.
    Keeps a bracket [u_low, u_high] with f(u_low) > target > f(u_high).
    A Newton step leaving the bracket is replaced by bisection in u,
    a step while the bracket is open is limited to max_step.
    Assumes target within the limits of the objective, see :func:`search`.
    If not converged within max_iter, returns the latest estimate,
    i.e., max_iter=1 is a single Taylor inversion around the initial guess.

    :param target: Desired value
    :type target: float
    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param objective: objective with derivative,
        payload-limited sender by default
    :type objective: :class:`Objective`
    :param lbda1: Initial guess for λ,
        1 / median of positive finite costs by default
    :type lbda1: tuple
    :param tol: tolerance for convergence
    :type tol: float
    :param max_iter: Maximum iterations
    :type max_iter: int
    :param max_step: maximal step in log λ while unbracketed
    :type max_step: float
    :return: Estimated λ value
    """
    objective = Objective.of(objective, e=e)
    if q is None:
        q = len(rhos) + 1
    #
    rhos = np.array(rhos, dtype=float)
    add_zero = True if q is None else len(rhos) == q-1

    # initial guess, scale-free
    lbda1 = None if lbda1 is None else np.ravel(lbda1)[0]
    if lbda1 is not None and lbda1 > 0:
        u = np.log(lbda1)
    else:
        rhos_pos = rhos[np.isfinite(rhos) & (rhos > 0)]
        u = -np.log(np.median(rhos_pos)) if rhos_pos.size > 0 else 0.
    u_low, u_high = -np.inf, np.inf
    log_target = np.log(target)
    #
    logging.debug(f'Newton started, {target=} lbda={np.exp(u)} {tol=} {objective=}')
    for it in range(max_iter):
        lbda = float(np.exp(u))
        ps = get_probability(rhos=rhos, lbda=lbda, add_zero=add_zero)

        f_val = objective(rhos=rhos, lbda=lbda, ps=ps, q=q)
        if abs(f_val - target) < tol:
            logging.debug(f'  {it=} | {lbda=:.03g} {f_val=:.03f} | {target=}')
            break

        # shrink bracket, f is decreasing in λ
        if f_val > target:
            u_low = u
        else:
            u_high = u

        # Newton step on g(u) = log f - log target, g'(u) = λ f' / f
        df_val = objective.derivative(rhos=rhos, lbda=lbda, ps=ps, q=q)
        logging.debug(f'  {it=} | {lbda=:.03g} {f_val=:.03f} {df_val=:.03f} | {target=}')
        if f_val > 0 and df_val < 0:
            u_newton = u - (np.log(f_val) - log_target) / (lbda * df_val / f_val)
        else:
            u_newton = np.nan
        bracketed = np.isfinite(u_low) and np.isfinite(u_high)
        if u_low < u_newton < u_high:
            if not bracketed:
                u_newton = u + np.clip(u_newton - u, -max_step, max_step)
            u = u_newton
        # fall back to bisection if Newton step leaves the bracket
        elif bracketed:
            u = .5 * (u_low + u_high)
        # expand bracket
        else:
            u = u + (max_step if f_val > target else -max_step)

    else:
        warnings.warn("Newton search for lambda did not converge", RuntimeWarning)
        # latest estimate, not yet evaluated
        lbda = float(np.exp(u))

    return lbda


def expsearch_bound(
    target: float,
    rhos: Tuple[np.ndarray],
    *,
    objective: Sender | Objective | Callable = Sender.PAYLOAD_LIMITED_SENDER,
    lbda0: Tuple[int] = None,
    max_iter: int = None,
    q: int = None,
    e: float = None,
) -> Tuple[Tuple[int], int]:
    rhos = np.array(rhos)
    objective = Objective.of(objective, e=e)
    if lbda0 is None:
        lbda0 = (-5, 15)
    #
    logging.debug(f'expsearch_bound started, {target=} {lbda0=} {objective=}')
    for i, k in enumerate(range(*lbda0)):
        #
        lbda1b = 10**k
        f_val = objective(rhos=rhos, lbda=lbda1b, q=q)
        logging.debug(f'  {k=} | {lbda1b=:.03g} {f_val=:.03f} | {target=}')

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
    SOLVER_BRENT = enum.auto()
    """Brent's method, without derivative."""

    @property
    def solve(self):
        solvers = {
            LambdaSolver.SOLVER_NEWTON: newton,
            LambdaSolver.SOLVER_BSEARCH: binary_search,
            LambdaSolver.SOLVER_BSEARCH_DDE: binary_search_dde,
            LambdaSolver.SOLVER_BRENT: brent,
        }
        return solvers[self]

    @property
    def expsearch(self):
        expsearch = {
            LambdaSolver.SOLVER_BSEARCH: expsearch_bound,
            # Newton, Brent and DDE search do their own bracketing
            LambdaSolver.SOLVER_NEWTON: None,
            LambdaSolver.SOLVER_BRENT: None,
            LambdaSolver.SOLVER_BSEARCH_DDE: None,
        }
        return expsearch[self]
