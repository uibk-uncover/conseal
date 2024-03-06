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
    if modify.lower() == Change.LSB_REPLACEMENT:
        rho_p1[cover % 2 != 0] = wet_cost
        rho_m1[cover % 2 == 0] = wet_cost

    # LSB matching
    elif modify.lower() == Change.LSB_MATCHING:
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

    >>> # TODO
    """
    # compute cost
    rho_p1, rho_m1 = compute_cost(
        cover=cover,
        modify=modify,
        wet_cost=wet_cost,
    )

    # sanitize range for LSB matching
    if modify.lower() == Change.LSB_MATCHING:
        cover_min, cover_max = cover_range
        where_min = cover == cover_min
        where_max = cover == cover_max
        rho_p1[where_min], rho_m1[where_min] = 0, rho_p1[where_min]
        rho_m1[where_max], rho_p1[where_max] = 0, rho_m1[where_max]

    return rho_p1, rho_m1


def probability(
    cover: np.ndarray,
    embedding_rate: float,
    *,
    modify: Change = Change.LSB_REPLACEMENT,
    permute: bool = True,
    cover_range: typing.Tuple[int] = (0, 255),
    cover_size: int = None,
    wet_cost: float = 10**10,
) -> np.ndarray:
    """Returns LSB probability map for consequent simulation.

    :param cover: cover image, in pixel or DCT domain, of arbitrary shape
    :type cover: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param embedding_rate: embedding rate,
        in bits per pixel
    :type embedding_rate: float
    :param modify: modification strategy, replacement by default
    :type modify: :class:`Change`
    :param permute: Permute the changes, otherwise sequential
    :type permute: bool
    :param cover_range: Range of cover values, (0,255) by default.
    :type cover_range: tuple
    :param cover_size: cover size, used for DCT cover, number of elements by default
    :type cover_size: int
    :param wet_cost: wet cost for unembeddable elements
    :type wet_cost: float
    :return: probability map of the same shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    e = 2  # embedding efficiency
    if cover_size is None:
        cover_size = cover.size

    p_p1 = np.zeros(cover.shape, dtype='double')
    p_m1 = np.zeros(cover.shape, dtype='double')
    if embedding_rate > 0:
        # number of changes
        involved_elements = int(np.ceil(embedding_rate * cover_size))
        changes = int(np.ceil(involved_elements / e))
        change_rate = changes / cover_size

        # permutative straddling
        if permute:
            p = np.ones(cover.shape, dtype=float) * change_rate
        # sequential embedding
        else:
            p = np.reshape(
                [1/e]*involved_elements + [0]*(cover_size - involved_elements),
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

    return p_p1, p_m1


def average_payload(*args, **kw):
    """Objective function for using LSB together with simulator.

    It sets a constant efficiency over all the embedding rates to 2.
    Other methods embed at the bound.
    """
    return simulate._ternary.average_payload(*args, e=2, **kw)
