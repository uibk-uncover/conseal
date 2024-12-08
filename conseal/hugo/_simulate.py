"""
Implementation of the HUGO steganography method as described in

T. Pevny, T. Filler, P. Bas
"Using High-Dimensional Image Models to Perform Highly Undetectable Steganography"
ACM Information Hiding 2010
https://hal.science/hal-00541353/document

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck

This implementation builds on the original Matlab implementation provided by the paper authors. Please find that license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2013 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501

import numpy as np

from . import _costmap
from ..simulate import _ternary


def simulate_single_channel(
    x0: np.ndarray,
    alpha: float,
    *,
    sigma: float = 1,
    gamma: float = 1,
    wet_cost: float = 1e8,
    **kw,
) -> np.ndarray:
    """Simulates HUGO embedding into a single channel.

    HUGO was introduced in
    T. Pevny, et al. Using High-Dimensional Image Models to Perform Highly Undetectable Steganography.
    ACM IH, 2010.

    The details of the methods are described in the
    `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#highly-undetectable-stego>`__.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
        in bits per pixel
    :type alpha: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :param kw: remaining keyword parameters are passed to simulator
    :type kw: dict
    :return: modified stego image
        of shape [height, width]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> x1 = cl.hugo.simulate_single_channel(
    ...     x0=x0,  # pixels
    ...     alpha=0.4,  # alpha
    ...     seed=12345)  # seed
    """
    #
    if alpha == 0:
        return x0

    # Compute cost
    rhos = _costmap.compute_cost_adjusted(
        x0,
        sigma=sigma,
        gamma=gamma,
        wet_cost=wet_cost,
    )

    # Simulate
    delta = _ternary.ternary(
        rhos=rhos,
        alpha=alpha,
        n=x0.size,
        **kw,
    )
    return x0 + delta.astype('uint8')
