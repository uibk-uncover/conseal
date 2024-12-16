"""
Implementation of simulated LSB steganography.

Unlike other methods, here we do not have a reference code,
and test visually and using known attacks and weaknesses of LSB.

Author: Martin Benes
Affiliation: University of Innsbruck
"""  # noqa: E501

import enum
import numpy as np
import typing

from .. import simulate
from .. import tools


class Change(enum.Enum):
    """Modify strategy for LSB steganography."""

    LSB_REPLACEMENT = enum.auto()
    """LSB replacement."""
    LSB_MATCHING = enum.auto()
    """LSB matching."""


def compute_cost(
    cover: np.ndarray,
    *,
    modify: Change = Change.LSB_REPLACEMENT,
    wet_cost: float = 10**10,
) -> np.ndarray:
    """Returns LSB cost.

    Provides unified interface with cost-based embeddings.

    :param cover: cover image, in pixel or DCT domain, of arbitrary shape
    :type cover: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param modify: modification strategy, replacement by default
    :type modify: :class:`Change`
    :param cover_range: Range of cover values, (0,255) by default.
    :type cover_range: tuple
    :param wet_cost: wet cost for unembeddable elements
    :type wet_cost: float
    :return: cost of the same shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    rho_p1 = np.ones(cover.shape)
    rho_m1 = np.ones(cover.shape)

    # LSB replacement
    if modify == Change.LSB_REPLACEMENT:
        rho_p1[cover % 2 != 0] = wet_cost
        rho_m1[cover % 2 == 0] = wet_cost

    # LSB matching
    elif modify == Change.LSB_MATCHING:
        pass

    else:
        raise NotImplementedError(
            f'unknown modify strategy {modify}'
        )

    return rho_p1, rho_m1


def compute_cost_adjusted(
    cover: np.ndarray,
    *,
    modify: Change = Change.LSB_REPLACEMENT,
    cover_range: typing.Tuple[int] = (0, 255),
    wet_cost: float = 10**10,
) -> np.ndarray:
    """Returns LSB distortion.

    Provides unified interface with distortion-based embeddings.

    :param cover: cover image, in pixel or DCT domain, of arbitrary shape
    :type cover: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param modify: modification strategy, replacement by default
    :type modify: :class:`Change`
    :param cover_range: Range of cover values, (0,255) by default.
    :type cover_range: tuple
    :param wet_cost: wet cost for unembeddable elements
    :type wet_cost: float
    :return: distortion of the same shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.lsb.compute_cost_adjusted(x0)
    >>> x1 = x0 + cl.simulate.ternary(rhos=rhos, alpha=.4, seed=12345)
    """
    # compute cost
    rho_p1, rho_m1 = compute_cost(
        cover=cover,
        modify=modify,
        wet_cost=wet_cost,
    )

    # sanitize range for LSB matching
    if modify == Change.LSB_MATCHING:
        cover_min, cover_max = cover_range
        where_min = cover == cover_min
        where_max = cover == cover_max
        rho_p1[where_min], rho_m1[where_min] = 0, rho_p1[where_min]
        rho_m1[where_max], rho_p1[where_max] = 0, rho_m1[where_max]

    return rho_p1, rho_m1


def probability(
    cover: np.ndarray,
    alpha: float,
    *,
    modify: Change = Change.LSB_REPLACEMENT,
    permute: bool = True,
    cover_range: typing.Tuple[int] = (0, 255),
    n: int = None,
    e: float = 2,
    wet_cost: float = 10**10,
) -> np.ndarray:
    """Returns LSB probability map for consequent simulation.

    :param cover: cover image
        of arbitrary shape
        in pixel or DCT domain
    :type cover: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
        in bits per pixel
    :type alpha: float
    :param modify: modification strategy, replacement by default
    :type modify: :class:`Change`
    :param permute: Permute the changes, otherwise sequential
    :type permute: bool
    :param cover_range: Range of cover values, (0,255) by default.
    :type cover_range: tuple
    :param n: cover size, used for DCT cover, number of elements by default
    :type n: int
    :param e: embedding efficiency
        in bits per change
    :type e: float
    :param wet_cost: wet cost for unembeddable elements
    :type wet_cost: float
    :return: probability map of the same shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> ps, _ = cl.lsb.probability(x0, alpha=.4)
    >>> x1 = x0 + cl.simulate._ternary.simulate(ps=ps, seed=12345)
    """
    if n is None:
        n = cover.size
    if e is None:
        e = alpha / tools.inv_entropy(alpha)

    p_p1 = np.zeros(cover.shape, dtype='double')
    p_m1 = np.zeros(cover.shape, dtype='double')
    if alpha > 0:
        # number of changes
        involved_elements = int(np.ceil(alpha * n))
        changes = int(np.ceil(involved_elements / e))
        change_rate = changes / n

        # permutative straddling
        if permute:
            p = np.ones(cover.shape, dtype=float) * change_rate
        # sequential embedding
        else:
            p = np.reshape(
                [1/e]*involved_elements + [0]*(n - involved_elements),
                cover.shape
            )

        # LSB replacement
        if modify == Change.LSB_REPLACEMENT:
            p_p1[cover % 2 == 0] = p[cover % 2 == 0]
            p_m1[cover % 2 != 0] = p[cover % 2 != 0]

        # LSB matching
        elif modify == Change.LSB_MATCHING:
            p_p1[:] = p/2
            p_m1[:] = p/2
            # sanitize min and max
            cover_min, cover_max = cover_range
            where_min = cover == cover_min
            where_max = cover == cover_max
            p_m1[where_min], p_p1[where_min] = 0, p[where_min]
            p_p1[where_max], p_m1[where_max] = 0, p[where_max]

        else:
            raise NotImplementedError(
                f'no modify strategy {modify}'
            )

    return (p_p1, p_m1), None


def average_payload(*args, **kw):
    """Objective function for using LSB together with simulator.

    It sets a constant efficiency over all the embedding rates to 2.
    Other methods embed at the bound.
    """
    return simulate._ternary.average_payload(*args, e=2, **kw)
