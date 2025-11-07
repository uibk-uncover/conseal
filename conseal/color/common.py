"""Tools for color image steganography.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import jpeglib
import numpy as np
from typing import Callable, Tuple

from .. import tools


def num_dct(
    jpeg: jpeglib.DCTJPEG
) -> int:
    """Calculates the number of elements in the JPEG.

    :param jpeg: Provided handle to the JPEG file.
    :type jpeg: `jpeglib.DCTJPEG <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.DCTJPEG>`_
    :returns: number of DCT coefficients
    :rtype: int

    :Example:

    >>> jpeg0 = jpeglib.read_dct('cover.jpeg')
    >>> n = cl.color.nc(jpeg0)
    """
    size = jpeg.Y.size
    if jpeg.has_chrominance:
        size += jpeg.Cb.size + jpeg.Cr.size
    if jpeg.has_black:
        size += jpeg.K.size
    return size


def map_channels(
    fun: Callable,
    x: np.ndarray,
    *args,
    stack_axis: int = -1,
    **kw,
) -> np.ndarray:
    """Calls function for each color channel (last axis).

    :param fun: Function to be called
    :type fun: Callable
    :param x: tensor of shape [..., channels]
    :type x: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param stack_axis: axis along which to stack the result, no stacking if None
    :type stack_axis: int
    :returns: processed tensor of shape [..., channels]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> x0 = np.array(Image.open('cover.png').convert('RGB'))
    >>> rhos =
    """
    x_ = [
        fun(x[..., c], *args, **kw)
        for c in range(x.shape[-1])
    ]
    if stack_axis is not None:
        return np.stack(x_, axis=stack_axis)
    else:
        return x_


def seed_per_channel(
    seed: int = None,
    num_channels: int = 3,
) -> Tuple[int]:
    """Converts single image seed to seeds for each color component.

    :param seed: random seed for the image
    :type seed: int
    :param num_channels: number of channels, 3 by default
    :type num_channels: int
    :return: array of per-channel seeds of shape [num_channels]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> image_seed = cl.tools.password_to_seed(Path(fname).stem)
    >>> channel_seeds = cl.color.seed_per_channel(
    ...     image_seed,
    ...     num_channels=x0.shape[-1])
    """
    gen = np.random.default_rng(seed)

    # Generate integers in the range [0, 2**31 - 1]
    return gen.integers(low=0, high=2 ** 31, dtype=np.int32, size=num_channels)


def transpose01(x):
    return tuple(map(list, zip(*x)))
