"""Simulator of binary embedding.

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck

This implementation builds on the Matlab implementation provided by the DDE lab. Please find that license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2013 DDE Lab, Binghamton University, NY. All Rights Reserved.
Permission to use, copy, modify, and distribute this software for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that this copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from DDE Lab. DDE Lab does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall Binghamton University or DDE Lab be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software. DDE Lab disclaims any warranties, and has no obligations to provide maintenance, support, updates, enhancements or modifications.
-------------------------------------------------------------------------
"""  # noqa: E501

import numpy as np
from typing import Tuple, Callable

from ._optim import get_objective, calc_lambda, Sender


def probability(
    rhos: Tuple[np.ndarray],
    alpha: float,
    n: int,
    *,
    e: float = None,
    objective: Callable = None,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
) -> Tuple[np.ndarray, float]:
    """Convert binary distortion to binary probability.

    :param rhos: distortion tensor for +-1 change
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate in bits per element, generally a constraint for the specified sender type
    :type alpha: float
    :param n: cover size/number of cover elements
    :type n: int
    :param e: Embedding efficiency,
        in bits per element.
        If not provided, optimal coding is assumed.
    :type e: float
    :param objective: Objective function to maximize.
        Entropy by default
    :type objective: callable
    :param sender: type of sender, changes the semantics of parameter alpha,
        PLS by default
    :return: tuple ((p_pm1), lmbda), where
        p_pm1 is the probability of +-1 change, and
        lbda is the determined lambda.
    :rtype: tuple

    :Example:

    >>> (p_pm1,), lbda = cl.simulate._binary.probability(
    ...   rhos=(rho_pm1,),  # costs for +1 and -1
    ...   alpha=0.4,        # embedding rate, 0.4 message bits per element
    ...   n=im_dct.Y.size)  # cover size
    >>> im_dct.Y += cl.simulate._binary.simulate(
    ...   ps=(p_pm1,),  # probability of +1 and -1
    ...   seed=12345)   # seed
    """
    if objective is None:
        objective = get_objective(e=e, q=2, sender=sender)

    message_length = int(np.round(alpha * n))
    lbda = calc_lambda(rhos, message_length, n, objective)
    #
    ps, _ = objective(lbda=lbda, rhos=rhos)
    return ps, lbda


def simulate(
    ps: Tuple[np.ndarray],
    generator: str = None,
    order: str = 'C',
    seed: int = None,
) -> np.ndarray:
    """Simulates changes using the given probability maps.

    :param ps: probability tensor for +-1 change,
        of an arbitrary shape
    :type p_p1: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param generator: random number generator to choose,
        None (numpy default) or 'MT19937' (used by Matlab)
    :type generator: str
    :param order: order of changes,
        'C' (C-order, column-row) or 'F' (F-order, row-column).
    :type order: str
    :param seed: random seed for embedding simulator
    :type seed: int
    :return: Simulated ternary changes in the cover, 0 (keep), +1 or -1.
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> (p_pm1,), lbda = cl.simulate._binary.probability(
    ...   rhos=(rho_pm1,),  # costs for +1 and -1
    ...   alpha=0.4,        # embedding rate, 0.4 message bits per element
    ...   n=im_dct.Y.size)  # cover size
    >>> im_dct.Y += cl.simulate._binary.simulate(
    ...   ps=(p_pm1,),  # probability of +1 and -1
    ...   seed=12345)   # seed
    """
    assert len(ps) in {1, 2}
    (p_pm1,) = ps[:1]

    # Select random number generator
    if generator is None:  # numpy default generator
        rng = np.random.default_rng(seed=seed)
        rand_change = rng.random(p_pm1.shape)

    elif generator == 'MT19937':  # Matlab generator
        prng = np.random.RandomState(seed)
        rand_change = prng.random_sample(p_pm1.shape)

    else:
        try:
            rand_change = generator(p_pm1.shape)
        except Exception:
            raise NotImplementedError(f'unsupported generator {generator}')

    # Order of changes
    if order is None or order == 'C':
        pass

    elif order == 'F':
        rand_change = rand_change.reshape(-1).reshape(p_pm1.shape, order='F')

    else:
        raise NotImplementedError(f'Given order {order} is not implemented')

    # Set up ndarray with simulated changes
    delta = np.zeros(p_pm1.shape, dtype='int8')

    delta[rand_change <= p_pm1] = 1
    return delta


def binary(
    rhos: Tuple[np.ndarray],
    alpha: float,
    n: int,
    *,
    e: float = None,
    objective: Callable = None,
    sender: Sender = Sender.PAYLOAD_LIMITED_SENDER,
    **kw,
) -> np.ndarray:
    """Simulates binary embedding given distortion and embedding rate.

    :param rhos: cost for +-1 changes,
        of an arbitrary shape
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per element
    :type alpha: float
    :param n: cover size/number of cover elements
    :type n: int
    :param objective: Objective function to maximize.
        Entropy by default
    :type objective: callable
    :return: Simulated difference image to be added to the cover, 0 (keep), 1 or -1 (change).
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> (rho_pm1,) = cl.uerd.compute_cost_adjusted(
    ...   cover_dct_coeffs=im_dct.Y,  # DCT
    ...   quantization_table=im_dct.qt[0])  # QT
    >>> im_dct.Y += cl.simulate.binary(
    ...   rhos=(rho_pm1,),  # cost of +-1 change
    ...   alpha=0.4,  # alpha
    ...   n=im_dct.Y.size,  # cover size
    ...   seed=12345)  # seed
    """
    ps, lbda = probability(rhos=rhos, alpha=alpha, n=n, sender=sender)
    return simulate(ps, **kw)
