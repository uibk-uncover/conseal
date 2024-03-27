"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum
import numpy as np


class Implementation(enum.Enum):
    """EBS implementation to choose from."""

    EBS_ORIGINAL = enum.auto()
    """Original EBS implementation by Remi Cogranne."""
    EBS_FIX_WET = enum.auto()
    """EBS implementation with wet-cost fixes in not-SI case."""


def _block_entropy(b: np.ndarray) -> float:
    """Computes block entropy of flattened block b."""
    b = b[1:]  # remove DC
    if (b == 0).all():
        return 0.
    values, counts = np.unique(b[b != 0], return_counts=True)
    Ps = counts.astype('float64') / np.sum(b != 0)
    H = -np.sum(Ps * np.log(Ps))  # mismatch with Matlab
    return H


def compute_cost(
    cover_dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
    rounding_error: np.ndarray = None,
    theta: float = 2,
) -> np.ndarray:
    """Compute EBS cost.

    From
    C. Wang, et al. An efficient JPEG steganographic scheme based on block entropy of DCT coefficients.
    IEEE ICASSP, 2012.

    Near-equivalent to Remi Cogranne's implementation.
    The difference comes from numerical difference in log/np.log.
    """
    # block entropy cost
    block_entropies = np.apply_along_axis(
        _block_entropy,
        axis=1,
        arr=cover_dct_coeffs.reshape(-1, 64),
    ).reshape(*cover_dct_coeffs.shape[:2], 1, 1)

    # block distortion
    block_distortion = 1/(block_entropies**theta)
    block_distortion = np.repeat(np.repeat(block_distortion, 8, 2), 8, 3)

    # flipping distortion
    if rounding_error is not None:
        # compute quantized DCT xbar_i = round(x_qi)
        # compute rounding error e_ri = xbar_i - x_qi
        # compute quantization error e_qi = xbar_i*qi - xi
        # compute flipping cost rho_flip_i = [(np.abs(y' - x_qi)-.5)*qi]**2
        raise NotImplementedError('distortion from precover not implemented')
    else:
        qti = quantization_table.copy()
        for i in range(2):
            qti = np.expand_dims(qti, i)
            qti = np.repeat(qti, cover_dct_coeffs.shape[i], i)
        flip_distortion = qti**2

    # final cost
    return block_distortion * flip_distortion


def compute_cost_adjusted(
    cover_dct_coeffs: np.ndarray,
    quantization_table: np.ndarray,
    precover: np.ndarray = None,
    theta: float = 2,
    implementation: Implementation = Implementation.EBS_ORIGINAL,
    wet_cost: float = 10**13,
) -> np.ndarray:
    """Computes the costmap and prepares the costmap for ternary embedding.

    :param cover_dct_coeffs: quantized cover DCT coefficients
        of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :type cover_dct_coeffs: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param quantization_table: quantization table
        of shape [8, 8]
    :type quantization_table: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param precover: precover for rounding error computation,
        of shape [num_vertical_blocks*8, num_horizontal_blocks*8]
    :param theta: distortion parameter, 2 by default (from paper)
    :type theta: float
    :param implementation: choose EBS implementation
    :type implementation: :class:`Implementation`
    :param wet_cost: cost for unembeddable coefficients
    :type wet_cost: float
    :return: probability maps for +1 and -1 changes,
        in DCT domain of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
    :rtype: tuple

    :Example:

    >>> rho_p1, rho_m1 = cl.ebs.compute_cost_adjusted(
    ...   cover_dct_coeffs=im_dct.Y,  # DCT
    ...   quantization_table=im_dct.qt[0])  # QT
    >>> im_dct.Y += cl.simulate.ternary(
    ...   rho_p1=rho_p1,  # distortion of +1
    ...   rho_m1=rho_m1,  # distortion of -1
    ...   alpha=0.4,  # alpha
    ...   n=im_dct.Y.size,  # cover size
    ...   seed=12345)  # seed
    """
    # Compute rounding error
    if precover is not None:
        raise NotImplementedError('side-informed EBS not implemented')
    else:
        rounding_error = None

    # Compute cost
    rho = compute_cost(
        cover_dct_coeffs=cover_dct_coeffs,
        quantization_table=quantization_table,
        rounding_error=rounding_error,
        theta=theta,
    )

    # Adjust embedding costs
    rho = rho + 10**-4

    # Assign wet cost
    rho[np.isinf(rho) | np.isnan(rho) | (rho > wet_cost)] = wet_cost

    # Do not embed into following modes (from Remi's implementation)
    if implementation == Implementation.EBS_ORIGINAL:
        rho[:, :, 0, 0] = wet_cost
        rho[:, :, 0, 4] = wet_cost
        rho[:, :, 4, 0] = wet_cost
        rho[:, :, 4, 4] = wet_cost

    elif implementation == Implementation.EBS_FIX_WET:
        # Avoid 04 coefficients with e = 0.5
        if precover is not None:
            max_cost_mat = np.zeros(shape=rho.shape, dtype=bool)
            max_cost_mat[:, :, 0, 0] = True
            max_cost_mat[:, :, 4, 0] = True
            max_cost_mat[:, :, 0, 4] = True
            max_cost_mat[:, :, 4, 4] = True
            rho[max_cost_mat & (np.abs(rounding_error) > 0.4999)] = wet_cost
    else:
        raise NotImplementedError(f'unknown implementation {implementation}')

    # # Do not embed +-1 if the DCT coefficient has extreme value
    # rho[cover_dct_coeffs >= 1023] = wet_cost
    # rho[cover_dct_coeffs <= -1023] = wet_cost
    # return rho

    # Do not embed +1 if the DCT coefficient has max value
    rho_p1 = np.copy(rho)
    rho_p1[cover_dct_coeffs >= 1023] = wet_cost

    # Do not embed -1 if the DCT coefficient has min value
    rho_m1 = np.copy(rho)
    rho_m1[cover_dct_coeffs <= -1023] = wet_cost

    return rho_p1, rho_m1
