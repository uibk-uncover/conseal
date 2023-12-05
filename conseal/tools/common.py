"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.optimize

EPS = np.finfo(np.float64).eps
"""small numerical constant"""


def _entropy(*ps) -> float:
    """Computes n-ary entropy.

    :param ps: Probability tensors.
    :type ps:
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
    px[px <= 0] = 1  # avoid log2(0)
    H = -(px*np.log2(px))
    return np.nansum(H)


def entropy(p: np.ndarray, p2: np.ndarray = None) -> float:
    """Computes entropy with complement probability added.

    :param p: Probability tensor.
    :type p: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param p2: Probability tensor.
    :type p2: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: Entropy
    :rtype: float

    :Example:

    >>> # TODO
    """
    if p2 is None:
        # Binary entropy
        return _entropy(p, 1-p)
    else:
        # Ternary entropy
        return _entropy(p, p2, 1-p-p2)


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
