"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import scipy.fftpack

from .common import entropy


def nzAC(dct: np.ndarray) -> int:
    """Computes number of non-zero DCT AC coefficients from 4D DCT tensor.

    :param dct: array of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type dct: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: number of non-zero DCT AC coefficients
    :rtype: int

    :Example:

    >>> cl.tools.nzAC(jpeg0.Y)
    """
    assert len(dct.shape) == 4, "Expected 4D DCT input array"
    return (dct != 0).sum() - (dct[:, :, 0, 0] != 0).sum()


def AC(dct: np.ndarray) -> int:
    """Computes number of DCT AC coefficients from 4D DCT tensor.

    :param dct: array of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type dct: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: number of DCT AC coefficients
    :rtype: int

    :Example:

    >>> cl.tools.AC(jpeg0.Y)
    """
    assert len(dct.shape) == 4, "Expected 4D DCT input array"
    return dct.size - np.prod(dct.shape[:2])


def embedding_rate(
    dct_c: np.ndarray,
    dct_s: np.ndarray,
    q: int = 3,
) -> float:
    """Estimate embedding rate in DCT domain.

    :param dct_c:
    :type dct_c: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param dct_s:
    :type dct_s: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param q:
    :type q: int
    :return:
    :rtype: float

    :Example:

    >>> # TODO
    """
    # no changes
    if (dct_c == dct_s).all():
        return 0.
    # embeddable elements
    embeddable = nzAC(dct_c)
    # binary bound
    if q == 2:
        # number of changes
        changes = (dct_c != dct_s).sum()
        # change rate
        betas = [changes/embeddable]

    # ternary bound
    elif q == 3:
        # number of changes
        changesP1 = (dct_c < dct_s).sum()
        changesM1 = (dct_c > dct_s).sum()
        # change rate
        betas = [
            changesP1 / embeddable,
            changesM1 / embeddable,
        ]

    else:
        raise NotImplementedError(f'{q}-ary code not implemented')

    # embedding rate
    alpha = entropy(*betas)
    return alpha


def Lambda(x: float) -> np.array:
    """DCT scaling function.

    :param x: Input value.
    :type x: float
    :return:
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:
    >>> Lambda(0) # -> 0.70710678
    >>> Lambda(-1) # -> 1.
    """
    return 1. if x != 0 else 1/np.sqrt(2)


def dct2(a: np.ndarray) -> np.ndarray:
    """"""
    a = scipy.fftpack.dct(a, axis=0, norm='ortho')
    a = scipy.fftpack.dct(a, axis=1, norm='ortho')
    return a


def idct2(a: np.ndarray) -> np.ndarray:
    """"""
    a = scipy.fftpack.idct(a, axis=0, norm='ortho')
    a = scipy.fftpack.idct(a, axis=1, norm='ortho')
    return a


def jpeglib_to_jpegio(dct_coeffs: np.ndarray) -> np.ndarray:
    """
    Convert DCT coefficient array shape from 4D jpeglib format to 2D jpegio format
    :param dct_coeffs: DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :return: DCT coefficients reshaped to [num_vertical_blocks * 8, num_horizontal_blocks * 8]

    :Example:

    >>> # TODO
    """
    num_vertical_blocks, num_horizontal_blocks, block_height, block_width = dct_coeffs.shape
    assert block_height == 8, "Expected block height of 8"
    assert block_width == 8, "Expected block width of 8"

    # Transpose from [num_vertical_blocks, num_horizontal_blocks, 8, 8] to [num_vertical_blocks, 8, num_horizontal_blocks, 8]
    dct_coeffs = dct_coeffs.transpose((0, 2, 1, 3))

    # Reshape to [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    dct_coeffs = dct_coeffs.reshape((num_vertical_blocks * 8, num_horizontal_blocks * 8))

    return dct_coeffs


def jpegio_to_jpeglib(dct_coeffs: np.ndarray) -> np.ndarray:
    """
    Convert DCT coefficient array shape from 2D jpegio format to 4D jpeglib format
    :param dct_coeffs: DCT coefficients of shape [num_vertical_blocks * 8, num_horizontal_blocks * 8]
    :return: DCT coefficients reshaped to [num_vertical_blocks, num_horizontal_blocks, 8, 8]

    :Example:

    >>> # TODO
    """
    # From jpegio 2D to jpeglib 4D
    assert dct_coeffs.shape[0] % 8 == 0
    assert dct_coeffs.shape[1] % 8 == 0

    num_vertical_blocks = dct_coeffs.shape[0] // 8
    num_horizontal_blocks = dct_coeffs.shape[1] // 8

    # Reshape from [num_vertical_blocks * 8, num_horizontal_blocks * 8] to [num_vertical_blocks, 8, num_horizontal_blocks, 8]
    dct_coeffs = dct_coeffs.reshape((num_vertical_blocks, 8, num_horizontal_blocks, 8))

    # Reorer to [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    dct_coeffs = dct_coeffs.transpose((0, 2, 1, 3))

    return dct_coeffs


def compute_DCT_mat() -> np.ndarray:
    """
    Computes the 8x8 DCT matrix

    :return: ndarray of shape [8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    [col, row] = np.meshgrid(range(8), range(8))
    dct_mat = 0.5 * np.cos(np.pi * (2 * col + 1) * row / (2 * 8))
    dct_mat[0, :] = dct_mat[0, :] / np.sqrt(2)
    return dct_mat


def block_dct2(
    spatial_blocks: np.ndarray,
    dct_mat: np.ndarray = None,
) -> np.ndarray:
    """
    Apply 2D DCT to image blocks
    :param spatial_blocks: ndarray of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :param dct_mat: ndarray of shape [8, 8]. If None, the DCT matrix is computed on-the-fly.
    :return: DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]

    :Example:

    >>> # TODO
    """
    if dct_mat is None:
        dct_mat = compute_DCT_mat()

    DCT_mat_left = dct_mat[None, None, :, :]
    DCT_mat_right = (dct_mat.T)[None, None, :, :]

    dct_coeffs = DCT_mat_left @ spatial_blocks @ DCT_mat_right

    return dct_coeffs


def block_idct2(
    dct_coeffs: np.ndarray,
    dct_mat: np.ndarray = None,
) -> np.ndarray:
    """Apply 2D inverse DCT to image blocks

    :param dct_coeffs: ndarray of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param dct_mat: ndarray of shape [8, 8]. If None, the DCT matrix is computed on-the-fly.
    :type dct_map: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: spatial blocks of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> # TODO
    """
    if dct_mat is None:
        dct_mat = compute_DCT_mat()

    DCT_mat_left = (dct_mat.T)[None, None, :, :]
    DCT_mat_right = dct_mat[None, None, :, :]

    spatial_blocks = DCT_mat_left @ dct_coeffs @ DCT_mat_right

    return spatial_blocks
