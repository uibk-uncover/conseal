"""
Implementation of λ estimation methods for the steganography simulator.

Based on the Gibbs distribution approach for minimal embedding distortion.
Provides several root‐finding strategies to solve E[D](λ) = target_distortion.

Author: Ilyas Satik, Martin Benes
Affiliation: University of Innsbruck
"""

import enum
import numpy as np
from typing import Tuple, Callable
import warnings

from ._defs import Sender
from .. import tools


def get_probability(
    rhos: Tuple[np.ndarray],
    lbda: float,
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
    # denominator (forced left-associativity)
    denum = 1 if add_zero else 0
    logits = []
    for rho in rhos:
        # print(rhos)
        logit = -lbda * rho
        logits.append(logit)
        denum += np.exp(logit)
    #
    denum[denum == 0] = tools.EPS
    # get probabilities
    ps = [
        np.exp(logit) / denum
        for logit in logits
    ]
    return ps, logits


def expected_distortion(
    rhos: np.ndarray,
    lbda: float,
    *,
    q: int = None,
    e: float = None,
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
    # calculate probabilities
    if q is None:
        add_zero = True
        q = len(rhos) + 1
    else:
        add_zero = len(rhos) == q-1
    # add_zero = True if q is None else len(rhos) == q-1
    ps, _ = get_probability(lbda=lbda, rhos=rhos, add_zero=add_zero)
    # expected distortion
    Erho = np.sum([
        rho * p
        for rho, p in zip(rhos, ps)
    ])
    return float(Erho)


def d_expected_distortion(
    rhos: np.ndarray,
    lbda: float,
    *,
    q: int = None,
    e: float = None,
) -> float:
    """
    Compute the derivative of expected distortion with respect to λ:
        dE[D]/dλ = −sum_i (rho_i^2 * p_i * (1 − p_i))

    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param lbda: inverse temperature λ
    :type lbda: float
    :return: value of the derivative of the expected distortion
    :rtype: float
    """
    # calculate probabilities
    if q is None:
        add_zero = True
        q = len(rhos) + 1
    else:
        add_zero = len(rhos) == q-1
    ps, logits = get_probability(lbda=lbda, rhos=rhos, add_zero=add_zero)
    # expected distortion
    if q == 2:
        dErho = -np.sum([
            (rho**2) * p * (1-p)
            for rho, p in zip(rhos, ps)
        ])
    elif q == 3:
        print(len(rhos), len(ps))
        # dErho = -np.sum([
        #     rho**2 * np.exp(-2 * log1pexp(logit) - log1pexp(-logit))
        #     for rho, logit in zip(rhos, logits)
        # ])
        dErho = -np.sum([
            (rho**2) * (p**2) * (1-p)
            for rho, p in zip(rhos, ps)
        ])

        # dErho = -np.sum(terms)
        # # log_ps = [np.log(p) for p in ps]
        # # dErho = -np.sum([
        # #     rho**2 * np.exp(2*np.log(p) + log1pexp(-p))
        # #     for rho, p in zip(rhos, ps)
        # # ])
        # ps = [np.clip(p, 1e-3, 1-1e-3) for p in ps]
        # rhos = [np.clip(rho, 1e-3, 1-1e-3) for rho in rhos]

        # H = -np.sum([
        #     p*logit - log1pexp(logit)
        #     for p, logit in zip(ps, logits)
        # ]) / np.log(2)
    else:
        raise NotImplementedError(f'unknown d_expected_distortion for {q=}')

    print(f'd_expected_distortion: {q=} {lbda=} {dErho=}')
    return float(dErho)


def log1pexp(x: np.ndarray) -> np.ndarray:
    """Numerically stable calculation of logarithm.

    Used for log(1+exp(L))=log(1-p).

    :param x: input argument
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: result
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    """
    out = np.empty_like(x)
    mask = x > 0
    out[mask] = x[mask] + np.log1p(np.exp(-x[mask]))
    out[~mask] = np.log1p(np.exp(x[~mask]))
    return out


def entropy_with_logit(
    ps: np.ndarray = None,
    logits: np.ndarray = None,
    *,
    e: float = None,
    q: float = None,
) -> float:
    """Compute the entropy H(p) of probability p.
    Logit is provided for better numerical stability.

    H = -sum p log2 p + (1-p) log2 (1-p)
    = -1/ln2 * sum (p L - log(1+exp(L)))

    :param p: change probability map
    :type p: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param logit: logit
    :type logit: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: value of the derivative of the entropy
    :rtype: float
    """
    # Imperfect coding - given embedding efficiency
    if e is not None:
        H = np.sum(ps) * e
    # Perfect coding - upper bound efficiency
    elif q == 2:
        H = -np.sum([
            p*logit - log1pexp(logit)
            for p, logit in zip(ps, logits)
        ]) / np.log(2)
    elif q == 3:
        # logit_p1, logit_m1 = logits
        # p_p1, p_m1 = ps
        # #
        # logZ = np.log1p(np.exp(logit_p1) + np.exp(logit_m1))  # log(1 + expLp + expLm)
        # H = -np.sum(p_p1 * logit_p1 + p_m1 * logit_m1 - logZ) / np.log(2)
        # return H
        # logZ = np.log1p(
        #     np.sum([np.exp(logit) for logit in logits], axis=0)
        # )
        # H = -np.sum([
        #     p * logit
        #     for p, logit in zip(ps, logits)
        # ])
        H = -np.sum([
            p*logit - log1pexp(logit)
            for p, logit in zip(ps, logits)
        ]) / np.log(2)
    else:
        raise NotImplementedError(f'unknown entropy for {q=}')

    return float(H)


def entropy(
    rhos: np.ndarray,
    lbda: float,
    *,
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
    if q is None:
        add_zero = True
        q = len(rhos) + 1
    else:
        add_zero = len(rhos) == q-1
    ps, logits = get_probability(rhos=rhos, lbda=lbda, add_zero=add_zero)
    return entropy_with_logit(ps=ps, logits=logits, q=q, e=e)


def entropy_dde(
    rhos: np.ndarray,
    lbda: float,
    *,
    q: int = None,
    e: float = None,
) -> float:
    if q is None:
        add_zero = True
        q = len(rhos) + 1
    else:
        add_zero = len(rhos) == q-1
    ps, logits = get_probability(rhos=rhos, lbda=lbda, add_zero=add_zero)

    # Imperfect coding - given embedding efficiency
    if e is not None:
        H = np.sum(ps) * e
    # Perfect coding - upper bound efficiency
    elif q is None or len(ps) == q-1:  # no change is zero cost
        H = tools.entropy(*ps)
    else:
        H = tools._entropy(*ps)

    return H


def d_entropy_with_logit(
    ps: np.ndarray,
    rhos: np.ndarray,
    logits: np.ndarray,
    *,
    q: int = 2,
    e: float = None,
) -> float:
    """Compute the derivative dH/dλ of the entropy H(p),
    where p is the probability map and λ is the inverse entropy.
    Logit is provided for better numerical stability.

    :param p: change probability map
    :type p: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho: embedding costs
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param logit: logit
    :type logit: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: value of the derivative of the entropy
    :rtype: float
    """
    # derivative dH/dλ
    # = 1/ln2 sum_i L_i p_i (1-p_i) rho_i
    assert e is None
    assert q-1 == len(ps)
    if q == 2:
        # dH = np.sum([
        #     logit * p * (1 - p) * rho
        #     for p, rho, logit in zip(ps, rhos, logits)
        # ]) / np.log(2)
        # print('Here!')
        dH = np.sum([
            logit * p * (1 - p) * rho
            for p, rho, logit in zip(ps, rhos, logits)
        ]) / np.log(2)
    elif q == 3:
        # print('Here!')

        logit_p1, logit_m1 = logits
        p_p1, p_m1 = ps
        rho_p1, rho_m1 = rhos
        # dH = -np.sum(
        #     rho_p1 * p_p1 * (1 - p_p1) +
        #     rho_m1 * p_m1 * (1 - p_m1) +
        #     (rho_p1 * p_p1 + rho_m1 * p_m1) * (p_p1 + p_m1)
        # ) / np.log(2)
        rho_bar = rho_p1 * p_p1 + rho_m1 * p_m1
        dH = np.sum(
            + p_p1 * (rho_p1 - rho_bar) * (1 + np.log(p_p1))
            + p_m1 * (rho_m1 - rho_bar) * (1 + np.log(p_m1))
            - (1 - p_p1 - p_m1) * rho_bar * (1 + np.log(1 - p_p1 - p_m1))
        ) / np.log(2)

        # dH = -np.sum([
        #     logit * p**2 * (1 - p) * rho
        #     for p, rho, logit in zip(ps, rhos, logits)
        # ]) / np.log(2)
        # dH = -np.sum([
        #     logit * p**2 * (1 - p) * rho
        #     for p, rho, logit in zip(ps, rhos, logits)
        # ]) / np.log(2)
    else:
        raise NotImplementedError(f'unknown d_entropy for {q=}')
    # dH = np.sum(logit * p * (1 - p) * rho) / np.log(2)
    # print(f'd_entropy_with_logit {dH=}')
    return float(dH)


def d_entropy(
    rhos: np.ndarray,
    lbda: float,
    *,
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
    if q is None:
        add_zero = True
        q = len(rhos) + 1
    else:
        add_zero = len(rhos) == q-1
    add_zero = True if q is None else len(rhos) == q-1
    ps, logits = get_probability(rhos=rhos, lbda=lbda, add_zero=add_zero)
    return d_entropy_with_logit(ps=ps, rhos=rhos, logits=logits, q=q, e=e)


def get_objective(
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    e: float = None,
) -> Callable:
    # assert e is None, 'e not implemented'
    if sender == Sender.PAYLOAD_LIMITED_SENDER:
        # print('using PLS H objective')
        if e is not None:
            return lambda *args, **kw: entropy(*args, e=e, **kw)
        else:
            return entropy
    elif sender == Sender.PAYLOAD_LIMITED_SENDER_DDE:
        if e is not None:
            return lambda *args, **kw: entropy_dde(*args, e=e, **kw)
        else:
            return entropy_dde
    elif sender == Sender.DISTORTION_LIMITED_SENDER:
        # print('using DiLS Erho objective')
        if e is not None:
            return lambda *args, **kw: expected_distortion(*args, e=e, **kw)
        else:
            return expected_distortion
    else:
        raise NotImplementedError(f'unknown sender {sender}')


def placeholder(*args, **kw):
    raise NotImplementedError


def get_d_objective(
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    e: float = None,
) -> Callable:
    # assert e is None, 'e not implemented'
    if sender == Sender.PAYLOAD_LIMITED_SENDER:
        # print('using PLS H d_objective')
        if e is not None:
            return placeholder
        else:
            return d_entropy
    elif sender == Sender.PAYLOAD_LIMITED_SENDER_DDE:
        if e is not None:
            return placeholder
        else:
            return d_entropy
    elif sender == Sender.DISTORTION_LIMITED_SENDER:
        # print('using DiLS Erho d_objective')
        if e is not None:
            return placeholder
        else:
            return d_expected_distortion
    else:
        raise NotImplementedError(f'unknown sender {sender}')


def binary_search(
    target: float,
    rhos: np.ndarray,
    objective: Callable,
    d_objective: Callable,
    *,
    lbda0: Tuple[float] = None,
    tol: float = 1e-3,
    n: int = None,
    max_iter: int = None,
    q: int = None,
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
    if lbda0 is None:
        lbda0 = (0, 1e4)
    if n is None:
        n = rhos[0].size
    if q is None:
        q = len(rhos) + 1
    if max_iter is None:
        max_iter = 30
    #
    low, high = lbda0
    # print()
    for it in range(max_iter):
        # if (high - low) <= tol:
        #     break
        mid = .5 * (low + high)
        f_val = objective(rhos=rhos, lbda=mid, q=q)
        # print(f'  {it=} | {mid=} {f_val=} {target=}')
        if abs(f_val - target) / n < tol:
            break
        if f_val > target:
            low = mid
        else:
            high = mid
    else:
        warnings.warn("optimization might not have converged", RuntimeWarning)
    lbda = .5 * (low + high)
    return get_probability(rhos=rhos, lbda=lbda)[0], lbda


def newton(
    target: float,
    rhos: np.ndarray,
    objective: Callable,
    d_objective: Callable,
    *,
    lbda0: Tuple[float] = None,
    tol: float = None,
    n: int = None,
    max_iter: int = None,
    q: int = None,
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
    if lbda0 is None:
        lbda0 = (target,)
    if n is None:
        n = rhos[0].size
    if q is None:
        q = len(rhos) + 1
    if tol is None:
        tol = 1e-3
    if max_iter is None:
        max_iter = 30
    #
    (lbda,) = lbda0
    #
    print(f'Newton started, {target=} {lbda0=} {tol=}')
    print(f'{objective=} {d_objective=}')
    for it in range(max_iter):
        f_val = objective(rhos=rhos, lbda=lbda)
        if abs(f_val - target) / n < tol:
            break
        df_val = d_objective(rhos=rhos, lbda=lbda)
        # update = (f_val - target) / df_val
        print(f'  {it=} | {lbda=} {f_val=} {df_val=} | {target=}')
        if df_val == 0:
            raise RuntimeError('derivative 0, cannot proceed')
        # lbda1 = lbda - update
        lbda1 = lbda - (f_val - target) / df_val
        if lbda1 < 0:
            raise RuntimeError('lambda below 0, wrong initialization')
        lbda = lbda1
    else:
        warnings.warn("optimization might not have converged", RuntimeWarning)
    return get_probability(rhos=rhos, lbda=lbda)[0], lbda


def polynomial_proxy(
    target: float,
    rhos: np.ndarray,
    objective: Callable,
    d_objective: Callable = None,
    *,
    lbda0: Tuple[float] = None,
    tol: float = None,
    n: int = None,
    max_iter: int = 15,
    q: int = None,
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
    if lbda0 is None:
        lbda = (5, 30)
    if n is None:
        n = rhos[0].size
    if q is None:
        q = len(rhos) - 1
    #
    xs = np.array(lbda0, dtype=np.float64)
    ys = np.array([
        objective(lbda=lbda, rhos=rhos)
        for lbda in lbda0
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
        lbda = xs[np.argmin([np.abs(ys - target) for lbda in lbda0])]
        # print(lbda)
    return get_probability(rhos=rhos, lbda=lbda)[0], lbda


def taylor_inverse(
    target: float,
    rhos: np.ndarray,
    objective: Callable,
    d_objective: Callable,
    *,
    lbda0: float = None,
    tol: float = None,
    n: int = None,
    max_iter: int = 15,
    q: int = None,
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
    if n is None:
        n = rhos[0].size
    if q is None:
        q = len(rhos) - 1
    if lbda0 is None:
        lbda0 = (5.,)
    # assess f(lbda) and f'(lbda)
    (lbda0,) = lbda0
    f_val = objective(lbda=lbda0, rhos=rhos)
    df_val = d_objective(lbda=lbda0, rhos=rhos)
    # calculate lambda from Taylor
    lbda = lbda0 + (target - f_val) / df_val
    #
    return get_probability(rhos=rhos, lbda=lbda)[0], lbda


def estimate_lambda_search(
    target: float,
    rhos: Tuple[np.ndarray],
    objective: Callable,
    d_objective: Callable,
    *,
    lbda0: Tuple[float] = None,
    tol: float = None,
    n: int = None,
    max_iter: int = None,
    q: int = None,
) -> Tuple[float, float]:
    assert n > 0, "Expected cover size greater than 0"
    if max_iter is None:
        max_iter = 15
    if lbda0 is None:
        lbda0 = (1000,)
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


def binary_search_dde(
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

    (l3, m3), iterations = estimate_lambda_search(
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
