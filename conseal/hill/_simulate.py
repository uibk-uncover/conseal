"""High-level interface of simulated HILL embedding.

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
    """Simulates HILL embedding into a single channel.

    HILL was introduced in
    B. Li, et al. A New Cost Function For Spatial Image Steganography.
    IEEE ICIP, 2014.

    The details of the methods are described in the
    `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#high-low-low>`__.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
        in bits per pixel
    :type alpha: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: stego pixels
        of shape [height, width, channels]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> x1 = cl.hill.simulate_single_channel(
    ...     x0=x0,  # cover pixels
    ...     alpha=.4,  # embedding rate [bpp]
    ...     seed=12345)  # seed for PSNR
    """
    if alpha == 0:
        return x0

    # Compute cost
    rhos = _costmap.compute_cost_adjusted(
        x0=x0,
        wet_cost=wet_cost,
    )

    # Simulate
    ps, _ = _ternary.probability(
        rhos=rhos,
        alpha=alpha,
        n=x0.size,
    )
    delta = _ternary.simulate(ps=ps, **kw)
    return x0 + delta.astype('uint8')
