"""Brent's method for the λ search, f(λ) = target.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import logging
import numpy as np
from typing import Tuple, Callable
import warnings

from ._common import Sender
from ._objective import Objective


def brent(
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
    """Brent's method to find λ satisfying f(λ) = target.

    Solves log f(λ) = log target in u = log λ, without derivative.
    Interpolates the inverse u(log f) through the last three points,
    i.e., inverse quadratic interpolation, or through two points (secant).
    Keeps a bracket [a, b] with the root inside.
    An interpolation leaving the bracket or shrinking it too slowly
    is replaced by bisection in u.
    The bracket is found by steps of max_step from the initial guess.
    Assumes target within the limits of the objective, see :func:`search`.
    If not converged within max_iter, returns the best estimate.

    :param target: Desired value
    :type target: float
    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param objective: objective, derivative is not needed,
        payload-limited sender by default
    :type objective: :class:`Objective`
    :param lbda1: Initial guess for λ,
        1 / median of positive finite costs by default
    :type lbda1: tuple
    :param tol: tolerance for convergence
    :type tol: float
    :param max_iter: Maximum number of objective evaluations
    :type max_iter: int
    :param max_step: step in log λ when searching for the bracket
    :type max_step: float
    :return: Estimated λ value
    """
    objective = Objective.of(objective, e=e)
    if q is None:
        q = len(rhos) + 1
    #
    rhos = np.array(rhos, dtype=float)
    log_target = np.log(target)

    def evaluate(u: float) -> Tuple[float, float, float]:
        """Returns λ, f(λ) and g(u) = log f - log target."""
        lbda = float(np.exp(u))
        f_val = objective(rhos=rhos, lbda=lbda, q=q)
        return lbda, f_val, np.log(max(f_val, np.finfo(float).tiny)) - log_target

    # initial guess, scale-free
    lbda1 = None if lbda1 is None else np.ravel(lbda1)[0]
    if lbda1 is not None and lbda1 > 0:
        u = np.log(lbda1)
    else:
        rhos_pos = rhos[np.isfinite(rhos) & (rhos > 0)]
        u = -np.log(np.median(rhos_pos)) if rhos_pos.size > 0 else 0.
    lbda, f_val, g_val = evaluate(u)
    it = 1
    logging.debug(f'Brent started, {target=} {lbda=} {tol=} {objective=}')
    if abs(f_val - target) < tol:
        return lbda

    # search for bracket, f is decreasing in λ
    a, g_a = u, g_val
    b, g_b = u, g_val
    while g_a < 0 and it < max_iter:
        a = a - max_step
        lbda, f_val, g_a = evaluate(a)
        it += 1
        logging.debug(f'  bracket {it=} | {lbda=:.03g} {f_val=:.03f} | {target=}')
        if abs(f_val - target) < tol:
            return lbda
    while g_b > 0 and it < max_iter:
        b = b + max_step
        lbda, f_val, g_b = evaluate(b)
        it += 1
        logging.debug(f'  bracket {it=} | {lbda=:.03g} {f_val=:.03f} | {target=}')
        if abs(f_val - target) < tol:
            return lbda
    # b is the best estimate so far
    if abs(g_a) < abs(g_b):
        a, b, g_a, g_b = b, a, g_b, g_a
    c, g_c = a, g_a
    d = None
    bisected = True

    while it < max_iter:
        # inverse interpolation of u(g) at g = 0
        if g_a != g_c and g_b != g_c:
            # inverse quadratic, through three points
            u = (
                a * g_b * g_c / ((g_a - g_b) * (g_a - g_c))
                + b * g_a * g_c / ((g_b - g_a) * (g_b - g_c))
                + c * g_a * g_b / ((g_c - g_a) * (g_c - g_b))
            )
        else:
            # secant, through two points
            u = b - g_b * (b - a) / (g_b - g_a)
        # fall back to bisection if interpolation leaves the bracket,
        # or does not shrink it at least by half of the previous step
        u_low, u_high = sorted(((3 * a + b) / 4, b))
        previous_step = abs(b - c) if bisected else abs(c - d)
        if not u_low < u < u_high or abs(u - b) >= .5 * previous_step:
            u = .5 * (a + b)
            bisected = True
        else:
            bisected = False

        lbda, f_val, g_val = evaluate(u)
        it += 1
        logging.debug(f'  {it=} | {lbda=:.03g} {f_val=:.03f} {bisected=} | {target=}')
        if abs(f_val - target) < tol:
            break

        # shrink bracket, keep b as the best estimate
        d, c, g_c = c, b, g_b
        if g_a * g_val < 0:
            b, g_b = u, g_val
        else:
            a, g_a = u, g_val
        if abs(g_a) < abs(g_b):
            a, b, g_a, g_b = b, a, g_b, g_a

    else:
        warnings.warn("Brent search for lambda did not converge", RuntimeWarning)
        # best estimate, not yet evaluated
        lbda = float(np.exp(b))

    return lbda


__all__ = ['brent']
