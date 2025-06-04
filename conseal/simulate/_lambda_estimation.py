"""
Implementation of λ estimation methods for distortion‐limited steganography simulator.

Based on the Gibbs distribution approach for minimal embedding distortion.
Provides several root‐finding strategies to solve E[D](λ) = target_distortion.

Author: Ilyas Satik
Affiliation: University of Innsbruck
"""

import numpy as np
import time
import typing
import math

# ------------------------------------------------------------------------------
# Synthetic Distortion Model Setup
# ------------------------------------------------------------------------------
np.random.seed(42)
_NUM_PIXELS = 5000

# Per‐pixel base and alternate distortion costs (binary embedding)
_rho0 = np.random.uniform(0.1, 1.0, size=_NUM_PIXELS)
_rho1 = np.random.uniform(0.1, 1.0, size=_NUM_PIXELS)
_delta = _rho1 - _rho0  # cost difference used in a logistic model


def expected_distortion(lmbda: float) -> float:
    """
    Compute the expected distortion E[D] under the Gibbs distribution for
    a binary embedding model:
        p_i = 1 / (1 + exp(lambda * delta_i))
        E[D] = sum_i (rho0_i + delta_i * p_i)

    :param lmbda: Inverse‐temperature parameter λ
    :return: Expected distortion value
    """
    p = 1.0 / (1.0 + np.exp(lmbda * _delta))
    return float(np.sum(_rho0 + _delta * p))


def d_expected_distortion(lmbda: float) -> float:
    """
    Compute the derivative of expected distortion with respect to λ:
        dE[D]/dλ = −sum_i (delta_i^2 * p_i * (1 − p_i))

    :param lmbda: Inverse‐temperature parameter λ
    :return: Derivative value
    """
    p = 1.0 / (1.0 + np.exp(lmbda * _delta))
    return float(-np.sum((_delta ** 2) * p * (1.0 - p)))


def binary_search_lambda(
        target: float,
        tol: float = 1e-6
) -> float:
    """
    Standard binary search to solve E[D](λ) = target up to tolerance.

    :param target: Desired expected distortion
    :param tol: Search tolerance
    :return: Estimated λ value
    """
    low, high = 0.0, 10.0
    while (high - low) > tol:
        mid = 0.5 * (low + high)
        if expected_distortion(mid) > target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def newton_lambda(
        target: float,
        initial: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 20
) -> float:
    """
    Newton‐Raphson method to find λ satisfying E[D](λ) = target.

    :param target: Desired expected distortion
    :param initial: Initial guess for λ
    :param tol: Tolerance for convergence
    :param max_iter: Maximum iterations
    :return: Estimated λ value
    """
    l = initial
    for _ in range(max_iter):
        f_val = expected_distortion(l) - target
        if abs(f_val) < tol:
            break
        df_val = d_expected_distortion(l)
        l -= f_val / df_val
    return l


def polynomial_proxy_lambda(
        target: float,
        samples: typing.Tuple[float, float, float] = (0.5, 2.0, 5.0)
) -> float:
    """
    Fit a quadratic polynomial y(λ) ≈ aλ² + bλ + c to sample points of
    (λ, E[D](λ)), then solve for λ where y(λ) = target.
    Fast single‐shot estimate; accuracy depends on model fit.

    :param target: Desired expected distortion
    :param samples: λ sample points for polynomial fit
    :return: Estimated λ value or NaN if no valid real root
    """
    xs = np.array(samples, dtype=np.float64)
    ys = np.array([expected_distortion(x) for x in xs], dtype=np.float64)
    a, b, c = np.polyfit(xs, ys, 2)
    roots = np.roots([a, b, c - target])
    real_roots = [r.real for r in roots if np.isreal(r) and (r.real > 0)]
    return min(real_roots) if real_roots else float('nan')


def taylor_inverse_lambda(
        target: float,
        lambda0: float = 1.0
) -> float:
    """
    Single‐step linear Taylor approximation: expand E[D](λ) around λ0, then invert.

    λ ≈ λ0 + (target − E[D](λ0)) / E’[D](λ0)

    :param target: Desired expected distortion
    :param lambda0: Expansion point for Taylor (default 1.0)
    :return: Estimated λ value (no iteration)
    """
    E0 = expected_distortion(lambda0)
    dE0 = d_expected_distortion(lambda0)
    return lambda0 + (target - E0) / dE0


def compare_methods() -> list:
    """
    Compare four λ‐estimation methods on a grid of target distortions:
      1) binary_search_lambda
      2) newton_lambda
      3) polynomial_proxy_lambda
      4) taylor_inverse_lambda

    Each method is timed (in ms) and its absolute error recorded against a
    high‐precision “true” λ given by binary_search_lambda(..., tol=1e-9).
    Returns a list of dicts with keys:
      'target_distortion', 'true_lambda',
      'binary_lambda', 'binary_time_ms', 'binary_error',
      'newton_lambda', 'newton_time_ms', 'newton_error',
      'poly_lambda', 'poly_time_ms', 'poly_error',
      'taylor_lambda','taylor_time_ms','taylor_error'
    """
    # Create five target‐distortions between E[D](0.1) and E[D](5.0)
    targets = np.linspace(
        expected_distortion(0.1),
        expected_distortion(5.0),
        num=5
    )
    # Use a tighter binary search as “true” λ
    true_lambdas = [binary_search_lambda(t, tol=1e-9) for t in targets]

    records = []
    for t, true_l in zip(targets, true_lambdas):
        rec = {'target_distortion': t, 'true_lambda': true_l}

        # 1) Binary Search (default tol=1e-6)
        t0 = time.perf_counter()
        bs = binary_search_lambda(t, tol=1e-6)
        rec['binary_lambda'] = bs
        rec['binary_time_ms'] = (time.perf_counter() - t0) * 1e3
        rec['binary_error'] = abs(bs - true_l)

        # 2) Newton’s Method
        t0 = time.perf_counter()
        nt = newton_lambda(t, initial=1.0)
        rec['newton_lambda'] = nt
        rec['newton_time_ms'] = (time.perf_counter() - t0) * 1e3
        rec['newton_error'] = abs(nt - true_l)

        # 3) Polynomial Proxy
        t0 = time.perf_counter()
        pp = polynomial_proxy_lambda(t)
        rec['poly_lambda'] = pp
        rec['poly_time_ms'] = (time.perf_counter() - t0) * 1e3
        rec['poly_error'] = (abs(pp - true_l) if not math.isnan(pp) else float('nan'))

        # 4) Taylor Inverse (linear around lambda0=1.0)
        t0 = time.perf_counter()
        tv = taylor_inverse_lambda(t, lambda0=1.0)
        rec['taylor_lambda'] = tv
        rec['taylor_time_ms'] = (time.perf_counter() - t0) * 1e3
        rec['taylor_error'] = abs(tv - true_l)

        records.append(rec)

    return records


if __name__ == "__main__":
    # If run directly, print out comparison results in key=value format
    results = compare_methods()
    for rec in results:
        line = ", ".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in rec.items()
        )
        print(line)
