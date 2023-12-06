"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck

This implementation builds on the Matlab implementation provided by the DDE lab. Please find that license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2013 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501

import numpy as np
import typing
import warnings

from .. import tools
from ._defs import get_p


def average_payload(
    lbda: float,
    rho_p1: np.ndarray = None,
    rho_m1: np.ndarray = None,
    p_p1: np.ndarray = None,
    p_m1: np.ndarray = None,
) -> float:
    """

    :param lbda:
    :type lbda: float
    :param rho_p1:
    :type rho_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_m1:
    :type rho_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param p_p1:
    :type p_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param p_m1:
    :type p_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return:
    :rtype: float

    :Example:

    >>> # TODO
    """
    assert (
        (p_p1 is not None and p_m1 is not None and rho_p1 is None and rho_m1 is None) or
        (p_p1 is None and p_m1 is None and rho_p1 is not None and rho_m1 is not None)
    ), 'exactly one of (p_p1, p_m1) or (rho_p1, rho_m1) must be given'

    if p_p1 is None:
        p_p1 = get_p(lbda, rho_p1, rho_m1)
        p_m1 = get_p(lbda, rho_m1, rho_p1)

    # Compute ternary entropy
    return (p_p1, p_m1), tools.entropy(p_p1, p_m1)


def calc_lambda(
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    message_length: int,
    n: int,
    objective: typing.Callable = None,
) -> float:
    """Implements binary search for lambda.

    The i-th element is embedded with a probability of
    p_i = 1/Z exp( -lambda D(X, y_i X_{~i}) ),
    where D is the distortion after modifying the i-th cover element, and Z is a normalization constant.
    This methods determines the lambda to communicate the message of the message length.

    We simulate a payload-limited sender that embeds a fixed average payload while minimizing the average distortion.
    Optimize for the lambda that minimizes the average distortion while transferring the message.

    :param rho_p1: Cost for embedding +1.
    :type rho_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_m1: Cost for embedding -1.
    :type rho_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param message_length: Message length.
    :type message_length: int
    :param n: Cover size.
    :type n: int
    :return: Parameter lambda value.
    :rtype: float

    :Example:

    >>> # TODO
    """
    assert n > 0, "Expected cover size greater than 0"

    # Initialize lambda and m3 such that the loop is at least entered once
    # m3 is the total entropy
    l3 = 1000
    m3 = float(message_length + 1)

    # Initialize iteration counter
    iterations = 0

    # Find the largest l3 such that the total entropy (m3) <= message_length
    while m3 > message_length:

        # Increase l3
        l3 *= 2

        # Compute total entropy m3
        _, m3 = objective(l3, rho_p1, rho_m1)  # objective function

        iterations += 1

        # unbounded = ternary search fails
        if iterations > 10:
            warnings.warn("unbounded distortion, ternary search fails", RuntimeWarning)
            return l3

    # Initialize lower bound to zero
    l1 = 0
    # The lower bound for the message size is n
    m1 = float(n)
    alpha = float(message_length) / n  # embedding rate

    # Binary search for lambda
    # Relative payload must be within 1e-3 of the required relative payload
    while float(m1 - m3) / n > alpha / 1000 and iterations < 30:
        # Mid of the interval [l1, l3]
        lbda = l1 + (l3 - l1) / 2

        # Calculate entropy at the mid of the interval
        _, m2 = objective(lbda, rho_p1, rho_m1)  # objective function

        # ternary search
        if m2 < message_length:
            # The average payload is too small for the message.
            # We need to decrease the upper bound
            l3 = lbda
            m3 = m2
        else:
            # The average payload exceeds the message length
            # We can increase the lower bound
            l1 = lbda
            m1 = m2

        # Proceed to the next iteration
        iterations = iterations + 1

    if iterations == 30:
        warnings.warn("optimization might not have converged", RuntimeWarning)

    return lbda


def probability(
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    alpha: float,
    n: int,
    objective: typing.Callable = None,
) -> np.ndarray:
    """Convert binary distortion to binary probability.

    :param rho_p1: distortion tensor for +1 change
    :type rho_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_m1: distortion tensor for -1 change
    :type rho_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
    :type alpha: float
    :param n: cover size
    :type n: int
    :return: tuple ((p_p1, p_m1), lmbda), where
        p_p1 is the probability of +1 change,
        p_m1 is the probability of -1 change, and
        lbda is the determined lambda.
    :rtype: tuple

    :Example:

    >>> (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
    ...   rho_p1=rho_p1,  # distortion of +1
    ...   rho_m1=rho_m1,  # distortion of -1
    ...   alpha=0.4,  # alpha
    ...   n=im_dct.Y.size)  # cover size
    >>> im_dct.Y += cl.simulate._ternary.simulate(
    ...   p_p1=p_p1,  # probability of +1
    ...   p_m1=p_m1,  # probability of -1
    ...   seed=12345)  # seed
    """
    if objective is None:
        objective = average_payload

    message_length = int(np.round(alpha * n))
    lbda = calc_lambda(rho_p1, rho_m1, message_length, n, objective)
    #
    (p_p1, p_m1), H = objective(lbda, rho_p1, rho_m1)
    return (p_p1, p_m1), lbda


def simulate(
    p_p1: np.ndarray,
    p_m1: np.ndarray,
    generator: str = None,
    order: str = 'C',
    seed: int = None,
) -> np.ndarray:
    """Simulates changes using the given probability maps.

    :param p_p1: probability tensor for +1 changes
        of an arbitrary shape
    :type p_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param p_m1: probability tensor for -1 changes
        of an arbitrary shape
    :type p_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param generator: random number generator to choose,
        None (numpy default) or 'MT19937' (used by Matlab)
    :type generator: str
    :param order: order of changes,
        'C' (C-order, column-row) or 'F' (F-order, row-column).
    :type order: str
    :param seed: random seed for embedding simulator
    :return: Simulated ternary changes in the cover, 0 (keep), +1 or -1.
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
    ...   rho_p1=rho_p1,  # distortion of +1
    ...   rho_m1=rho_m1,  # distortion of -1
    ...   alpha=0.4,  # alpha
    ...   n=im_dct.Y.size)  # cover size
    >>> im_dct.Y += cl.simulate._ternary.simulate(
    ...   p_p1=p_p1,  # probability of +1
    ...   p_m1=p_m1,  # probability of -1
    ...   seed=12345)  # seed
    """

    # Select random number generator
    if generator is None:  # numpy default generator
        rng = np.random.default_rng(seed=seed)
        rand_change = rng.random(p_p1.shape)

    elif generator == 'MT19937':  # Matlab generator
        prng = np.random.RandomState(seed)
        rand_change = prng.random_sample(p_p1.shape)

    else:
        raise NotImplementedError(f'unsupported generator {generator}')

    # Order of changes
    if order is None or order == 'C':
        pass

    elif order == 'F':
        rand_change = rand_change.reshape(-1).reshape(p_p1.shape, order='F')

    else:
        raise NotImplementedError(f'Given order {order} is not implemented')

    # Set up ndarray with simulated changes
    delta = np.zeros(p_p1.shape, dtype='int8')

    # rand_change < p_p1 => increment 1
    # rand_change < p_p1 + p_m1 => decrement 1

    delta[rand_change < p_p1] = 1
    delta[(rand_change >= p_p1) & (rand_change < p_p1+p_m1)] = -1
    return delta


def ternary(
    rho_p1: np.ndarray,
    rho_m1: np.ndarray,
    alpha: float,
    n: int,
    **kw,
) -> np.ndarray:
    """Simulates ternary embedding given distortion and embedding rate.

    :param rho_p1: distortion tensor for +1 changes
        of an arbitrary shape
    :type rho_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_m1: distortion tensor for -1 changes
        of an arbitrary shape
    :type rho_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param n: cover size
    :type n: int
    :return: Simulated difference image to be added to the cover, 0 (keep), 1 or -1 (change).
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rho_p1, rho_m1 = cl.uerd.compute_distortion(
    ...   cover_dct_coeffs=im_dct.Y,  # DCT
    ...   quantization_table=im_dct.qt[0])  # QT
    >>> im_dct.Y += cl.simulate.ternary(
    ...   rho_p1=rho_p1,  # distortion of +1
    ...   rho_m1=rho_m1,  # distortion of -1
    ...   alpha=0.4,  # alpha
    ...   n=im_dct.Y.size,  # cover size
    ...   seed=12345)  # seed
    """
    (p_p1, p_m1), lbda = probability(
        rho_p1=rho_p1,
        rho_m1=rho_m1,
        alpha=alpha,
        n=n,
    )
    return simulate(p_p1, p_m1, **kw)
