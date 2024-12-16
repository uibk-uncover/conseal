"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum
import numpy as np
import scipy.signal
from typing import Tuple

from .. import mipod
from .common import EPS


class Attacker(enum.Enum):
    """Type of atacker."""

    ATTACKER_OMNISCIENT = enum.auto()
    """Omniscient attacker with SCA."""
    ATTACKER_INDIFFERENT = enum.auto()
    """Indifferent attacker without SCA."""


def estimate_variance(
    x: np.ndarray,
) -> np.ndarray:
    """
    Estimates the pixels' variance in 3x3 pixel neighborhood as

    V[X] = E[X^2] - (E[X])^2

    :param x: image
    :type x: np.ndarray
    :param block_size: size of the local neighborhood, denoted as p in the paper. See Fig. 2.
        Small p: extreme content adaptivity;
        Medium p: Medium content adaptivity;
        Large p: Low content adaptivity.
    :type block_size: int
    :param degree: degree of the polynomial
    :type degree: int
    :return: estimated variance per pixel
    :rtype:
    """
    """"""
    x = x.astype('float32')

    # local mean
    kernel = np.ones((3, 3), dtype='float')
    R = scipy.signal.convolve2d(np.ones_like(x), kernel, mode='same', boundary='fill')
    Ex = scipy.signal.convolve2d(x, kernel, mode='same', boundary='fill') / R

    # local variance
    Ex2 = scipy.signal.convolve2d(x**2, kernel, mode='same', boundary='fill') / R
    Vx = Ex2 - Ex**2
    return Vx


def attack(
    x0: np.ndarray,
    ps: Tuple[np.ndarray],
    *,
    clip: float = EPS,
    attacker: Attacker = Attacker.ATTACKER_OMNISCIENT,
) -> float:
    """Likelihood ratio test with cover x0 and assumed change rates.

    The method assumes a Gaussian cover model.
    In practice, it is used to quickly benchmark steganographic methods,
    without need to train an actual classifier.

    It was introduced in
    Sedighi, Cogranne, Fridrich.
    Content-Adaptive Steganography by Minimizing Statistical Detectability.
    IEEE TIFS, 2016.

    :param x0: uncompressed (pixel) cover image
        of shape [height, width]
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps: probability tensors for changes
        of an arbitrary shape
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param clip: bottom clip of variance to ensure numerical stability,
        EPS by default
    :type clip: float
    :param attacker: assumed attacker,
        omniscient attacker (knowing SCA) by default
    :type attacker: :class:`Attacker`
    :return: value of the deflection coefficient
    :rtype: float

    :Example:

    >>> rho_p1, rho_m1 = cl.hill.compute_cost_adjusted(x0)
    >>> (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
    ...     rhos=(rho_p1, rho_m1),
    ...     alpha=.1,
    ...     n=x0.size)
    >>> defl_hill = cl.tools.lrt.attack(x0, (p_p1, p_m1), clip=1e-3)
    """
    assert len(x0.shape) == 2, 'grayscale expected'
    x0 = x0.astype('float32')
    # ternary to binary
    p = np.sum(ps, axis=0)

    # estimate variance (with MiPOD estimator)
    # Vx0 = estimate_variance(x0)  # worse
    v0 = x0 - mipod.wiener2(x0, kernel_size=(2, 2))
    Vx0 = mipod.estimate_variance(v0, block_size=3, degree=3)

    # fisher information
    Vx0 = np.clip(Vx0, clip, None)  # clip FI in flat areas
    fi = 1. / Vx0**2
    fi[np.isinf(fi)] = 0  # ignores infinities

    # deflection coefficient
    # Eq. 11
    if attacker == Attacker.ATTACKER_OMNISCIENT:
        lr = np.sqrt(2) * np.sqrt(np.nansum(fi * p**2))
    # Eq. 12
    elif attacker == Attacker.ATTACKER_INDIFFERENT:
        lr = np.sqrt(2) * np.nansum(fi * p) / np.sqrt(np.nansum(fi))
    else:
        raise NotImplementedError(f'unknown attacker {attacker}')
    return float(lr)
