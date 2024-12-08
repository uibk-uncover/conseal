"""
J-MiPOD implementation based on https://codeocean.com/capsule/7800700/tree/v4
Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck

This Python implementation is ported from the original Matlab implementation provided by the paper authors. Please find the license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2020 Remi Cogranne, UTT (Troyes University of Technology). All Rights Reserved.
-------------------------------------------------------------------------
This code is provided by the author under Creative Common License (CC BY-NC-SA 4.0) which, as explained on this webpage https://creativecommons.org/licenses/by-nc-sa/4.0/ Allows modification, redistribution, provided that:
* You share your code under the same license;
* That you give credits to the authors;
* The code is used only for non-commercial purposes (which includes education and research)
-------------------------------------------------------------------------
The authors hereby grant the use of the present code without fee, and without a written agreement under compliance with aforementioned and provided and the present copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from the UTT or the authors. The UTT does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall the UTT or the authors be liable to any party for any consequential damages. The authors also fordid any practical use of this code for communication by hiding data into JPEG images.
-------------------------------------------------------------------------
"""

import os.path
import numpy as np
import scipy.io
import scipy.optimize
import scipy.signal
from typing import Tuple
import warnings

from .. import tools


def wiener2(
    im: np.ndarray,
    kernel_size: Tuple[int],
    *,
    noise: float = None,
) -> np.ndarray:
    """
    Adaptive noise removal.

    Uses zero padding to match the Matlab implementation.

    From the Matlab docstring:
    Wiener lowpass-filters an intensity image that has been degraded by constant power additive noise.
    This filter uses a pixel-wise adaptive Wiener method based on statistics estimated from a local neighborhood of each pixel.

    :param im: 2D ndarray
    :type im: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param kernel_size: 2-tuple containing the window size for estimating local mean and standard deviation
    :type kernel_size: tuple of int
    :param noise: power of assumed noise
    :type noise: float
    :return: filtered image
    :rtype:
    """

    # Parse given kernel size
    if kernel_size is None:
        # Default to isotropic kernel of size 3
        kernel_size = [3] * im.dim
    kernel_size = np.asarray(kernel_size)
    if kernel_size.shape == ():
        # Expand integer kernel size to the image dimensions
        kernel_size = np.repeat(kernel_size.item(), im.ndim)

    num_kernel_elements = np.prod(kernel_size)

    # Estimate the local mean.
    # The Matlab implementation uses zero-padding.
    local_mean = scipy.signal.correlate2d(im, np.ones(kernel_size), mode="same", boundary="fill") / num_kernel_elements

    # Estimate the local variance
    local_mean_squared = scipy.signal.correlate2d(im ** 2, np.ones(kernel_size), mode="same", boundary="fill") / num_kernel_elements
    local_variance = local_mean_squared - local_mean ** 2
    # FIX: for flat images
    # if np.isnan(local_variance).any():
    #     warnings.warn('invalid variance in flat areas, clipping')
    #     local_variance[np.isnan(local_variance)] = 0
    #     local_variance = np.clip(local_variance, a_min=0.01, a_max=None)
    if (local_variance == 0).any():
        warnings.warn('invalid variance in flat areas, clipping')
        local_variance = np.clip(local_variance, a_min=1e-8, a_max=None)

    # Estimate the noise power if needed
    if noise is None:
        noise = np.mean(local_variance)

    res = im - local_mean
    res *= (1 - noise / np.maximum(local_variance, noise))  # Avoid division by zero variance. If local variance < noise, we are going to replace it with the local mean later anyway.
    res += local_mean
    out = np.where(local_variance < noise, local_mean, res)

    # Sanity check
    out2 = local_mean + (np.maximum(0, local_variance - noise) / np.maximum(local_variance, noise)) * (im - local_mean)
    assert np.allclose(out, out2)

    return out


def im2col(
    img: np.ndarray,
    window_shape: Tuple[int],
    *,
    step_size: int = 1,
) -> np.ndarray:
    """
    Return sliding window over the given 2D image
    :param img: input image
    :type img: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param window_shape: 2-tuple with the window size
    :type window_shape:
    :param step_size: step size
    :type step_size: int
    :return: view of shape [window_height, window_width, num_windows]
    """
    # Determine number of output windows
    img_height, img_width = img.shape
    num_rows = img_height - window_shape[0] + 1
    num_cols = img_width - window_shape[1] + 1
    shape = (window_shape[0], window_shape[1], num_rows, num_cols)

    s0, s1 = img.strides
    strides = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return out_view.reshape((window_shape[0] * window_shape[1], num_rows * num_cols))[:, ::step_size]


def pad_copy_borders(x, pad_size):
    pad_top, pad_bottom = pad_size[0]
    pad_left, pad_right = pad_size[1]

    # Zero-padding leaves the original array in the center.
    x_padded = np.pad(x, pad_width=pad_size, mode="constant")

    # Fill the top rows by copying from the next 8 rows
    if pad_top > 0:
        x_padded[:pad_top, :] = x_padded[pad_top:2 * pad_top, :]

    # Fill the bottom rows
    if pad_bottom > 0:
        x_padded[-pad_bottom:, :] = x_padded[-2 * pad_bottom:-pad_bottom, :]

    # Fill left
    if pad_left > 0:
        x_padded[:, :pad_left] = x_padded[:, pad_left:2 * pad_left]

    # Fill right
    if pad_right > 0:
        x_padded[:, -pad_right:] = x_padded[:, -2 * pad_right:-pad_right]

    return x_padded


def invxlnx3_fast(y, f, excess_values_num_iter=10):
    """
    Fast solver of y = x * log(x - 2) parallelized over all pixels
    :param y: pixels?
    :param f: lookup table for solving y = x * log(x - 2) for x, with resolution 100
    :param excess_values_num_iter: number of iterations for computing the values that exceed the precomputed range
    :return: x
    """

    # The lookup table only contains values up to y = 1000.
    i_large = y > 1000
    i_small = y <= 1000

    # The lookup table uses a finer resolution with values y = [0, 1 / resolution, 2 / resolution, ..., 1000].
    # The default resolution is 100.
    iyL = np.floor(y[i_small] / 0.01).astype(int)
    iyR = iyL + 1

    # Clamp values that would exceed the lookup table
    iyR[iyR > 100000] = 100000

    x = np.zeros_like(y)
    x[i_small] = f[iyL] + (y[i_small] - iyL * 0.01) * (f[iyR] - f[iyL])

    # For values outside the precomputed range, we iteratively compute the result.
    z = y[i_large] / np.log(y[i_large] - 2)
    for _ in range(excess_values_num_iter):
        z = y[i_large] / np.log(z - 2)

    x[i_large] = z

    return x


def ternary_entropy(probs):
    return tools.entropy(probs, probs, base=np.e)

# def ternary_entropy(probs):
#     p0 = 1 - 2 * probs
#     P = np.hstack([p0, probs, probs])
#     H = -P * np.log(P)
#     #
#     eps = np.finfo(np.float32).eps
#     H[P < eps] = 0
#     #
#     return np.nansum(H)


def ternary_probs(
    fisher_information: float,
    payload: int,
    max_num_iterations: int = 30,
    excess_values_num_iter: int = 10,
) -> float:
    """
    Determine the change rates for each cover element by searching for the Lagrangian multiplier lambda that optimizes the payload constraint.
    :param fisher_information: fisher information
    :type fisher_information: float
    :param payload: target payload
    :type payload: int
    :param max_num_iterations: maximum number of binary search iterations
    :type max_num_iterations: int
    :param excess_values_num_iter: number of iterations for computing the values that exceed the precomputed range
    :type excess_values_num_iter: int
    :return: change rate per cover element
    :rtype: float
    """
    ixlnx3 = load_lookup_table()

    fisher_information_flat = fisher_information.flatten(order="F")

    # Initial search interval for lambda
    L = 10 ** 3
    R = 10 ** 6

    # fL, fR > 0 when they fit the desired payload
    # Note that smaller lambdas lead to higher ternary entropy.
    fL = ternary_entropy(1. / invxlnx3_fast(L * fisher_information_flat, ixlnx3, excess_values_num_iter=excess_values_num_iter)) - payload
    fR = ternary_entropy(1. / invxlnx3_fast(R * fisher_information_flat, ixlnx3, excess_values_num_iter=excess_values_num_iter)) - payload

    # If the range [L, R] does not cover the desired payload, enlarge the search interval.
    # fL < 0 and fR < 0: smallest lambda is still too big, expand search interval to the left by halving L
    # fL > 0 and fR > 0: highest lambda is still too low, expand search interval to the right by doubling R
    while fL * fR > 0:
        if fL > 0:
            R = 2 * R
            fR = ternary_entropy(1. / invxlnx3_fast(R * fisher_information_flat, ixlnx3, excess_values_num_iter=excess_values_num_iter)) - payload
        else:
            L = L / 2
            fL = ternary_entropy(1. / invxlnx3_fast(L * fisher_information_flat, ixlnx3, excess_values_num_iter=excess_values_num_iter)) - payload

    # At this point, L is below and R is above the desired lambda. Hence, fL > 0 and fR < 0.

    # Search for lambda (variable M) in the specified interval
    i = 0

    # fM is the distance between the target payload and the payload with the current lambda
    fM = 1

    # Record lambda and the distance between the target payload and the current payload
    TM = np.zeros((max_num_iterations, 2))

    # Binary search for lambda
    # Stopping criteria:
    # (1) The distance between the target payload and the current payload is less than or equal to 0.0001.
    # (2) Stop after a maximum of 30 iterations.
    while np.abs(fM) > 0.0001 and i < max_num_iterations:
        # Mid of the interval
        M = (L + R) / 2

        # Evaluate distance between target payload and current payload
        fM = ternary_entropy(1. / invxlnx3_fast(M * fisher_information_flat, ixlnx3, excess_values_num_iter=excess_values_num_iter)) - payload

        # fL is > 0 by construction
        if fL * fM < 0:
            # If the current lambda fits the payload, then we can cap the search interval at the top.
            R = M
            fR = fM

        else:
            # fM > 0: We can cap the search interval at the bottom.
            L = M
            fL = fM

        # Store current results
        TM[i, 0] = fM
        TM[i, 1] = M

        # Increment iteration counter
        i += 1

    if max_num_iterations == i:
        # Maximum number of iterations reached

        # Find the lambda that achieved the lowest absolute distance to the target payload
        M = TM[np.argmin(np.abs(TM[:, 0])), 1]

    # Compute beta using the lambda found
    # beta is the change rate
    beta = 1. / invxlnx3_fast(M * fisher_information_flat, ixlnx3, excess_values_num_iter=excess_values_num_iter)

    return beta


def prepare_lookup_table(max_val=1000, resolution=100, ftol=1e-3):
    """
    Prepare a lookup table for inverting the function y = x * log(x - 2).

    The values in the lookup table correspond to y in [0, max_value].
    The lookup table entries resolve with finer resolution, i.e., [0, 1 / resolution, 2 / resolution, ..., max_value].

    If you want to know which x gives the value y, do

    x = lut[ floor(y * resolution) ]
    x * log(x - 2) should be close to y

    :param max_val: highest y value in the lookup table (inclusive)
    :param resolution: precision of the lookup table, should be >= 1.
    :param ftol: tolerance in function value; stopping criterion of the minimum search
    :return: lookup table of the desired size
    """

    def fun(x):
        return x * np.log(x - 2)

    num_entries = max_val * resolution + 1
    xs = np.zeros(num_entries)

    for i in range(num_entries):
        y = i / resolution

        # We want to minimize the absolute distance between y and f(x)
        def find_inverse(x):
            return np.abs(y - fun(x))

        x = scipy.optimize.fmin(func=find_inverse, x0=3, xtol=1e-6, ftol=ftol, disp=0)[0]

        # Quick sanity check
        assert np.isclose(fun(x), y, atol=ftol)

        # Store the result
        xs[i] = x

    return xs


def load_lookup_table():
    mat_file = os.path.join(os.path.dirname(__file__), "ixlnx3.mat")
    npy_file = os.path.join(os.path.dirname(__file__), "ixlnx3.npz")

    if os.path.exists(mat_file):
        # Loading lookup table from Matlab
        ixlnx3 = scipy.io.loadmat(mat_file)["ixlnx3"].flatten()

    elif os.path.exists(npy_file):
        # Loading our own pre-computed lookup table
        ixlnx3 = np.load(npy_file)["ixlnx3"]

    else:
        # No lookup table yet
        print("Lookup table does not exist yet. Preparing and storing lookup table. This can take a few minutes.")

        # Need to create the lookup table
        ixlnx3 = prepare_lookup_table(max_val=1000, resolution=100)

        # Store lookup table
        np.savez(npy_file, ixlnx3=ixlnx3)

    return ixlnx3
