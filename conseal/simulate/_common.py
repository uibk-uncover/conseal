"""Common definitions of the simulate module.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum
import numpy as np
from typing import Tuple


class Sender(enum.Enum):
    """Type of sender."""

    PAYLOAD_LIMITED_SENDER = enum.auto()
    """Payload-limited sender."""
    DISTORTION_LIMITED_SENDER = enum.auto()
    """Distortion-limited sender."""


def get_probability(
    rhos: np.ndarray | Tuple[np.ndarray],
    lbda: float,
    *,
    add_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts distortions into probabilities,
    using Boltzmann-Gibbs distribution

    For more details, see `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#embedding-simulation>`__.

    :param rhos: distortion of embedding choices, e.g. embedding +1 or embedding -1
    :type rhos: tuple
    :param lbda: parameter value
    :type lbda: float
    :param add_zero:
    :type add_zero: bool
    :param p_pm1: probability tensor for changes associated to rhos[0]
        of an arbitrary shape
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    rhos = np.array(rhos)
    exponent = -lbda * rhos

    # Log-Sum-Exp stabilization
    exponent_max = np.max(exponent, axis=0, keepdims=True)
    if add_zero:
        exponent_max = np.maximum(0.0, exponent_max)
    exponent_stable = exponent - exponent_max
    gibbs = np.exp(exponent_stable)

    #
    denum = np.exp(-exponent_max) if add_zero else 0.0
    return gibbs / np.clip(denum + np.nansum(gibbs, axis=0, keepdims=True), 1e-15, None)


def _finite_rhos(rhos: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Zeroes costs of impossible states, avoids 0 * inf = nan for wet costs."""
    return np.where(ps > 0, rhos, 0.)


def _variance(rhos: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Per-element variance of cost under Gibbs, the zero state contributes ρ_0 = 0."""
    rhos = _finite_rhos(rhos, ps)
    return np.sum(ps * rhos**2, axis=0) - np.sum(ps * rhos, axis=0)**2
