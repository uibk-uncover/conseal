"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import typing

from . import _costmap
from ._costmap import Change, Location
from ..simulate import _ternary


def simulate(
    cover: np.ndarray,
    alpha: float,
    *,
    modify: Change = Change.LSB_REPLACEMENT,
    locate: Location = None,
    permute: bool = None,  # deprecated
    cover_range: typing.Tuple[int] = (0, 255),
    n: int = None,
    e: float = 2,
    rhos: np.ndarray = None,
    wet_cost: float = 10**10,
    **kw,
) -> np.ndarray:
    """Simulates LSB steganography at an embedding rate into a cover,
    and returns stego.

    Allows both replacement and matching, sequential and permuted pass.
    LSB is well-known since early 1990's.

    The details of the methods are described in the
    `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#least-significant-bit>`__.

    :param x0: cover image, in pixel or DCT domain,
        of arbitrary shape
    :type x0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
        in bits per element
    :type alpha: float
    :param modify: modification strategy,
        replacement by default
    :type modify: :class:`Change`
    :param permute: permute the changes, otherwise sequential
    :type permute: bool
    :param cover_range: range of cover values,
        (0,255) by default
    :type cover_range: tuple
    :param n: cover size, used for DCT cover, number of elements by default
    :type n: int
    :param e: embedding efficiency
        in bits per change
    :type e: float
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :param kw: remaining parameters passed to simulator
    :type kw: dict
    :return: stego image of the same meaning and shape as cover
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> x0 = np.array(Image.open('cover.pgm'))
    >>> x1 = cl.lsb.simulate(
    ...   cover=x0,
    ...   alpha=0.4,
    ...   modify=cl.LSB_MATCHING,
    ...   seed=12345)
    >>> Image.fromarray(x1).save('stego.pgm')

    The function can also take DCT coefficients

    >>> jpeg0 = jpeglib.read_dct('cover.jpeg')
    >>> jpeg1 = jpeg0.copy()
    >>> jpeg1.Y = cl.lsb.simulate(
    ...   cover=jpeg0.Y,
    ...   alpha=0.4,
    ...   modify=cl.LSB_MATCHING,
    ...   cover_range=(-1024, 1023),
    ...   seed=12345)
    >>> jpeg1.write_dct('stego.jpeg')

    Technically, the function can also simulate LSB for color images.
    This is the reason, why it is called `simulate` instead of `simulate_single_channel`.

    >>> x0 = np.array(Image.open('cover.ppm'))
    >>> x1 = cl.lsb.simulate(
    ...   cover=x0,
    ...   alpha=0.4,
    ...   modify=cl.LSB_MATCHING,
    ...   seed=12345)
    >>> Image.fromarray(x1).save('stego.ppm')
    """
    # compute probability
    ps, _ = _costmap.probability(
        cover=cover,
        alpha=alpha,
        modify=modify,
        locate=locate,
        permute=permute,
        cover_range=cover_range,
        n=n,
        e=e,
        rhos=rhos,
        wet_cost=wet_cost
    )
    # simulate
    delta = _ternary.simulate(
        ps=ps,
        **kw,
    )
    return cover + delta.astype(cover.dtype)
