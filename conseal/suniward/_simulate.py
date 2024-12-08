"""
High-level interface of simulated S-UNIWARD embedding.


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
    sigma: float = 1,
    wet_cost: float = 10 ** 8,
    **kw,
) -> np.ndarray:
    """Simulates S-UNIWARD embedding into a single channel.

    :param x0: uncompressed (pixel) cover image,
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per pixel
    :type alpha: float
    :param sigma: stabilizing constant. Small sigmas make the embedding very sensitive to the image content. Large sigmas smooth out the embedding change probabilities.
    :type sigma: float
    :param wet_cost: cost for unembeddable coefficients
    :type wet_cost: float
    :return: stego pixels,
        of shape [height, width, channels]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> x1 = cl.suniward.simulate_single_channel(
    ...   x0=x0,  # pixels
    ...   alpha=0.4,  # embedding rate
    ...   seed=12345)  # seed
    """
    if alpha == 0:
        return x0

    # compute distortion
    rhos = _costmap.compute_cost_adjusted(
        x0=x0,
        sigma=sigma,
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

