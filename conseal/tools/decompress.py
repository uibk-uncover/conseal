import numpy as np
from .dct import block_idct2


def decompress_channel(y, qt=None):
    """Decompress a single channel.

    :param y: quantized DCT coefficients,
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :param qt: quantization table,
        of shape [8, 8]
    :return: decompressed channel,
        of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8].
        The values should be in the range [0, 255], although some values may exceed this range.
    """

    # Validate quantization table
    if qt is not None:
        assert (
                len(qt.shape) == 2
                and qt.shape[0] == 8
                and qt.shape[1] == 8), "Expected quantization table of shape [8, 8]"

    num_vertical_blocks, num_horizontal_blocks = y.shape[:2]

    # If quantization table is given, dequantize the DCT coefficients
    if qt is not None:
        y = y * qt[None, None, :, :]

    # Inverse DCT
    x = block_idct2(y)

    # Reorder blocks to obtain image
    x = np.transpose(x, axes=[0, 2, 1, 3]).reshape(num_vertical_blocks * 8, num_horizontal_blocks * 8)

    # Level shift
    x += 128

    return x
