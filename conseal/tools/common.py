"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.optimize

EPS = np.finfo(np.float64).eps
"""small numerical constant"""


def _entropy(*ps, base: float = 2) -> float:
    """Computes n-ary entropy.

    :param ps: Probability tensors.
    :type ps:
    :param base:
    :type base:
    :param base: unit of information, bits by default
    :type base: float
    :return: Entropy.
    :rtype: float

    :Example:

    >>> # TODO
    """
    # probabilities
    px = np.array(list(ps))
    assert np.all((-EPS <= px) & (px <= 1+EPS)), 'invalid probabilities'
    assert np.isclose(np.sum(px, axis=0), 1).all(), 'denormalized probabilities'
    # entropy
    px[px <= 0] = 1  # avoid log(0)
    H = -(px * np.log(px) / np.log(base))
    return np.nansum(H)


def entropy(*ps, base: int = 2):
    """Computes entropy with complement probability added.

    :param p: Probability tensor.
    :type p: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param base: unit of information, bits by default
    :type base: float
    :return: Entropy
    :rtype: float

    :Example:

    >>> # TODO
    """
    px = np.array(list(ps))
    px0 = 1-np.sum(px, axis=0)
    return _entropy(*px, px0, base=base)

# def entropy(p: np.ndarray, p2: np.ndarray = None) -> float:
#     """Computes entropy with complement probability added.

#     :param p: Probability tensor.
#     :type p: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
#     :param p2: Probability tensor.
#     :type p2: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
#     :return: Entropy
#     :rtype: float

#     :Example:

#     >>> # TODO
#     """
#     if p2 is None:
#         # Binary entropy
#         return _entropy(p, 1-p)
#     else:
#         # Ternary entropy
#         return _entropy(p, p2, 1-p-p2)


def inv_entropy(
    h: float,
) -> float:
    """Computes inverse entropy.

    :param h:
    :type h:
    :return:
    :rtype: float

    :Example:

    >>> # TODO
    """
    global EPS

    def mse_H_eq_h(p) -> float:
        """SS error from H(p)=h."""
        return (_entropy(p, 1-p) - h)**2

    return scipy.optimize.fminbound(mse_H_eq_h, 0, .5, xtol=1e-4)


def matlab_round(x):
    """
    Helper method to mimic Matlab rounding
    |x| >= 0.5 => round away from zero
    :param x: ndarray
    :return: rounded ndarray
    """

    # Preserve the input sign
    x_sign = np.sign(x)

    # Convert to absolute value
    x_abs = np.abs(x)

    # Round towards zero
    x_abs_floor = np.floor(x_abs)

    # Round away from zero
    x_abs_ceil = np.ceil(x_abs)

    # Compute difference between value and floored value
    abs_floor_diff = x_abs - x_abs_floor

    # Condition for rounding away from zero
    mask_ceil = (abs_floor_diff >= 0.5) | np.isclose(abs_floor_diff, 0.5)

    # Ceil or floor
    x_rounded = np.where(mask_ceil, x_abs_ceil, x_abs_floor)

    # Restore sign
    x_rounded *= x_sign

    return x_rounded
