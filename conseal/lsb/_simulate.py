"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import typing

from . import _costmap
from ._costmap import Change
from ..simulate import _ternary


def simulate(
    cover: np.ndarray,
    embedding_rate: float,
    *,
    modify: Change = Change.LSB_REPLACEMENT,
    permute: bool = True,
    cover_range: typing.Tuple[int] = (0, 255),
    cover_size: int = None,
    wet_cost: float = 10**10,
    **kw,
) -> np.ndarray:
    """Simulates LSB steganography at an embedding rate into a cover,
    and returns stego.

    Allows both replacement and matching, sequential and permuted pass.
    LSB is well-known since early 1990's.

    The details of the methods are described in the
    `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#least-significant-bit>`__.

    :param cover: cover image, in pixel or DCT domain, of arbitrary shape
    :type cover: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param embedding_rate: embedding rate,
        in bits per element
    :type embedding_rate: float
    :param modify: modification strategy, replacement by default
    :type modify: :class:`Change`
    :param permute: Permute the changes, otherwise sequential
    :type permute: bool
    :param cover_range: Range of cover values, (0,255) by default.
    :type cover_range: tuple
    :param cover_size: cover size, used for DCT cover, number of elements by default
    :type cover_size: int
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :param kw: remaining parameters passed to simulator
    :type kw: dict
    :return: stego image of the same meaning and shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> x = np.array(Image.open('cover.pgm'))
    >>> y = cl.lsb.simulate_single_channel(
    ...   cover_spatial=x,
    ...   embedding_rate=0.4,
    ...   modify=cl.LSB_MATCHING,
    ...   seed=12345)
    >>> Image.fromarray(y).save('stego.pgm')
    """
    # compute probability
    p_p1, p_m1 = _costmap.probability(
        cover=cover,
        embedding_rate=embedding_rate,
        modify=modify,
        permute=permute,
        cover_range=cover_range,
        cover_size=cover_size,
        wet_cost=wet_cost
    )
    # rho_p1, rho_m1 = _costmap.compute_distortion(
    #     modify=modify,
    #     cover_range=cover_range,
    #     wet_cost=wet_cost,
    # )
    # simulate
    delta = _ternary.simulate(
        p_p1=p_p1,
        p_m1=p_m1,
        **kw,
    )
    return cover + delta  # .astype('uint8')
