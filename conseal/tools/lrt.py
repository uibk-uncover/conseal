"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum
import numpy as np
import scipy.signal
from typing import Tuple

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

    Moved to sealwatch, use :func:`sealwatch.mvg.deflection` instead.

    :raises NotImplementedError: always
    """
    raise NotImplementedError(
        'cl.tools.lrt.attack() moved to sealwatch, use sw.mvg.deflection() instead'
    )
