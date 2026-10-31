"""
Implementation of the nsF5 steganography method as described in

J. Fridrich, T. Pevny, and J. Kodovsky.
"Statistically undetectable JPEG steganography: Dead ends, challenges, and opportunities"
Multimedia & Security, 2007
http://dde.binghamton.edu/kodovsky/pdf/Fri07-ACM.pdf

Author: Martin Benes
Affiliation: University of Innsbruck
"""  # noqa: E501

import numpy as np

from .. import coding
from ._costmap import compute_cost


def embed_single_channel(
    y0: np.ndarray,
    message: bytes,
    key: int,
    *,
    block_size: int = None,
    wet_cost: float = 10**10,
) -> np.ndarray:
    """Embeds message with nsF5 into single-channel cover and returns stego.

    Decreases absolute values of nonzero AC coefficients,
    selected by wet-paper code, to carry the message in their parity.
    Zeros and DC are wet.
    Coefficients shrunk to 0 still carry the message,
    so the receiver does not need to know about them.

    Uses wet-paper code with the Block Minimal method, as the original nsF5,
    see :func:`conseal.coding.wpc.embed`.

    :param y0: quantized cover DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y0: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param message: message to embed
    :type message: bytes
    :param key: stego key, shared with the receiver
    :type key: int
    :param block_size: number of message bits per block,
        18 by default, see :func:`conseal.coding.wpc.embed`
    :type block_size: int
    :param wet_cost: wet cost for unembeddable coefficients
    :type wet_cost: float
    :return: quantized stego DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :raises ValueError: if the message does not fit

    :Example:

    >>> jpeg1.Y = cl.nsF5.embed_single_channel(
    ...   y0=jpeg0.Y,         # DCT
    ...   message=b'secret',  # message
    ...   key=12345)          # stego key
    """
    assert len(y0.shape) == 4, 'DCT must have 4 dimensions'
    assert y0.shape[2] == y0.shape[3] == 8, 'expected blocks of size 8x8'

    # costs, wet for zeros and DC
    rho = compute_cost(y0, wet_cost=wet_cost)

    # keyed random order of coefficients,
    # hides the message and spreads wet coefficients over the blocks
    order = np.random.default_rng(key).permutation(y0.size)

    # binary wet-paper code on parity of the coefficients
    changes = np.zeros(y0.size, dtype='uint8')
    changes[order] = coding.wpc.embed(
        (y0 % 2).flatten()[order], rho.flatten()[order], message,
        wet_cost=wet_cost,
        block_size=block_size,
    )

    # flip parity by decreasing the absolute value
    return (y0 - np.sign(y0) * changes.reshape(y0.shape)).astype(y0.dtype)


def extract_single_channel(
    y1: np.ndarray,
    length: int,
    key: int,
    *,
    block_size: int = None,
) -> bytes:
    """Extracts message embedded with nsF5 from single-channel stego.

    :param y1: quantized stego DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type y1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param length: message length in bytes
    :type length: int
    :param key: stego key, shared with the sender
    :type key: int
    :param block_size: number of message bits per block,
        as in :func:`embed_single_channel`
    :type block_size: int
    :return: extracted message
    :rtype: bytes

    :Example:

    >>> message = cl.nsF5.extract_single_channel(
    ...   y1=jpeg1.Y,  # DCT
    ...   length=6,    # message length in bytes
    ...   key=12345)   # stego key
    """
    assert len(y1.shape) == 4, 'DCT must have 4 dimensions'
    assert y1.shape[2] == y1.shape[3] == 8, 'expected blocks of size 8x8'

    # same order as the sender
    order = np.random.default_rng(key).permutation(y1.size)

    # parity of all coefficients, shrunk zeros included
    return coding.wpc.extract(
        (y1 % 2).flatten()[order], length,
        q=2, block_size=block_size,
    )
