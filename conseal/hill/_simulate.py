"""
High-level interface of simulated HILL embedding.


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
    wet_cost: float = 10**10,
    **kw,
) -> np.ndarray:
    """Simulate HILL embedding into a single channel.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per pixel
    :type alpha: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: stego pixels,
        of shape [height, width, channels]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    if alpha == 0:
        return x0

    # Compute distortion
    rhos = _costmap.compute_cost_adjusted(
        x0=x0,
        wet_cost=wet_cost,
    )

    # simulator
    ps, _ = _ternary.probability(
        rhos=rhos,
        alpha=alpha,
        n=x0.size,
    )
    delta = _ternary.simulate(ps=ps, **kw)

    return x0 + delta.astype('uint8')
