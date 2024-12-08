"""
High-level interface of simulated WOW embedding.


Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np

from . import _costmap
from ..simulate import _ternary


def simulate_single_channel(
    x0: np.ndarray,
    alpha: float,
    *,
    p: float = -1,
    wet_cost: float = 10 ** 8,
    **kw,
) -> np.ndarray:
    """Simulates WOW embedding into a single channel.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per pixel
    :type alpha: float
    :param p: parameter for reciprocal Hoelder norm.
    :type p: float
    :param wet_cost: cost for unembeddable coefficients
    :type wet_cost: float
    :return: stego pixels,
        of shape [height, width, channels]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> x1 = cl.wow.simulate_single_channel(
    ...   x0=x0,  # pixels
    ...   alpha=0.4,  # embedding rate
    ...   seed=12345)  # seed
    """
    if alpha == 0:
        return x0

    # compute distortion
    rhos = _costmap.compute_cost_adjusted(
        x0=x0,
        p=p,
        wet_cost=wet_cost
    )

    # simulator
    delta = _ternary.ternary(
        rhos=rhos,
        alpha=alpha,
        n=np.prod(x0.shape),
        **kw,
    )

    return x0 + delta.astype('uint8')
