import numpy as np
from .dct import block_idct2


def decompress_channel(dct_coeffs, quantization_table=None):
    """
    Decompress a single channel
    :param dct_coeffs: quantized DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :param quantization_table: quantization table of shape [8, 8]
    :return: decompressed channel of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8]. The values should be in the range [0, 255], although some values may exceed this range.
    """

    # Validate quantization table
    if quantization_table is not None:
        assert (
                len(quantization_table.shape) == 2
                and quantization_table.shape[0] == 8
                and quantization_table.shape[1] == 8), "Expected quantization table of shape [8, 8]"

    num_vertical_blocks, num_horizontal_blocks = dct_coeffs.shape[:2]

    # If quantization table is given, unquantize the DCT coefficients
    if quantization_table is not None:
        dct_coeffs = dct_coeffs * quantization_table[None, None, :, :]

    # Inverse DCT
    spatial_blocks = block_idct2(dct_coeffs)

    # Reorder blocks to obtain image
    spatial = np.transpose(spatial_blocks, axes=[0, 2, 1, 3]).reshape(num_vertical_blocks * 8, num_horizontal_blocks * 8)

    # Level shift
    spatial += 128

    return spatial
