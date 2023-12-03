
import numpy as np
import scipy.optimize

EPS = np.finfo(np.float64).eps
"""small numerical constant"""


def _entropy(*ps):
    """Computes n-ary entropy.

    Args:
        ps (*): Change probabilities.
    Returns:
        (float): Entropy.
    """
    # probabilities
    px = np.array(list(ps))
    assert np.all((-EPS <= px) & (px <= 1+EPS)), 'invalid probabilities'
    assert np.isclose(np.sum(px, axis=0), 1).all(), 'denormalized probabilities'

    # entropy
    px[px <= 0] = 1  # avoid log2(0)
    H = -(px*np.log2(px))
    return np.nansum(H)


def entropy(p, p2=None):
    """Computes entropy with complement probability added.

    Args:
        p (*): Change probabilities.
    Returns:
        (float): Binary entropy.
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

    Args:
        h (float): Entropy.
    Returns:
        (float): Probability.
    """
    global EPS

    def mse_H_eq_h(p) -> float:
        """SS error from H(p)=h."""
        return (_entropy(p, 1-p) - h)**2

    return scipy.optimize.fminbound(mse_H_eq_h, 0, .5, xtol=1e-4)

    # return scipy.optimize.minimize_scalar(
    #     mse_H_eq_h,
    #     method='Bounded',
    #     bounds=(0, 1)
    # ).x
