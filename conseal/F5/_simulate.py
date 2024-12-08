"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np

from .. import nsF5


def simulate_single_channel(*args, **kw) -> np.ndarray:
    """Simulates F5 embedding at an embedding rate at an embedding rate into single-channel cover and returns stego.

    This is done by simulating nsF5 first, and then re-introducing the shrinkage.

    :return: quantized stego DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y = cl.F5.simulate_single_channel(
    ...   y0=im_dct.Y,  # DCT
    ...   alpha=0.4,  # alpha
    ...   seed=12345)  # seed
    """
    return nsF5.simulate_single_channel(*args, add_shrinkage=True, **kw)
