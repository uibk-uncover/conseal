"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import typing
import warnings

from .. import tools
from ._defs import get_p


def average_payload(
    lbda: float,
    rhoP1: np.ndarray = None,
    rhoM1: np.ndarray = None,
    pP1: np.ndarray = None,
    pM1: np.ndarray = None,
) -> float:
    assert (
        (pP1 is not None and pM1 is not None and rhoP1 is None and rhoM1 is None) or
        (pP1 is None and pM1 is None and rhoP1 is not None and rhoM1 is not None)
    ), 'exactly one of (pP1, pM1) or (rhoP1, rhoM1) must be given'

    if pP1 is None:
        pP1 = get_p(lbda, rhoP1, rhoM1)
        pM1 = get_p(lbda, rhoM1, rhoP1)

    # Compute ternary entropy
    return (pP1, pM1), tools.entropy(pP1, pM1)


def calc_lambda(
    rhoP1: np.ndarray,
    rhoM1: np.ndarray,
    message_length: int,
    n: int,
    objective: typing.Callable = None,
):
    """Implements binary search for lambda.

    The i-th element is embedded with a probability of
    p_i = 1/Z exp( -lambda D(X, y_i X_{~i}) ),
    where D is the distortion after modifying the i-th cover element, and Z is a normalization constant.
    This methods determines the lambda to communicate the message of the message length.

    We simulate a payload-limited sender that embeds a fixed average payload while minimizing the average distortion.
    Optimize for the lambda that minimizes the average distortion while transferring the message.

    Args:
        rhoP1 (np.ndarray): cost for embedding +1.
        rhoM1 (np.ndarray): cost for embedding -1.
        message_length (int): Message length.
        n (int): Cover size.
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
        _, m3 = objective(l3, rhoP1, rhoM1)  # objective function

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
        _, m2 = objective(lbda, rhoP1, rhoM1)  # objective function

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
    rhoP1: np.ndarray,
    rhoM1: np.ndarray,
    alpha: float,
    n: int,
    objective: typing.Callable = None,
) -> np.ndarray:
    """Convert binary distortion to binary probability.

    Args:
        rhoP1 (np.ndarray): Distortion tensor for +1 change.
        rhoM1 (np.ndarray): Distortion tensor for -1 change.
        alpha (float): Embedding rate.
        n (int): Embeddable elements.
    Returns:
        2-tuple (pChangeP1, pChangeM1), lbda
            pChangeP1 is the probability of embedding +1
            pChangeM1 is the probability of embedding -1
            lbda is the determined lambda
    """
    if objective is None:
        objective = average_payload

    message_length = int(np.round(alpha * n))
    lbda = calc_lambda(rhoP1, rhoM1, message_length, n, objective)
    #
    (pP1, pM1), H = objective(lbda, rhoP1, rhoM1)
    return (pP1, pM1), lbda


def simulate(
    pChangeP1: np.ndarray,
    pChangeM1: np.ndarray,
    generator: str = None,
    order: str = 'C',
    seed: int = None,
):
    """
    Simulates changes using the given probability maps.

    Args:
        pChangeP1 (np.ndarray): Probability tensor for changes +1.
        pChangeM1 (np.ndarray): Probability tensor for changes -1.
        generator (str): Random number generator to choose. One of None (numpy default), or MT19937, used by Matlab.
        order (str): Order of changes, C or F.
        seed (int): Random number generator seed for reproducibility.
    Returns:
        (np.ndarray): Simulated ternary changes in the cover, 0 (keep), +1 or -1.
    """

    # Select random number generator
    if generator is None:  # numpy default generator
        rng = np.random.default_rng(seed=seed)
        rand_change = rng.random(pChangeP1.shape)

    elif generator == 'MT19937':  # Matlab generator
        prng = np.random.RandomState(seed)
        rand_change = prng.random_sample(pChangeP1.shape)

    else:
        raise NotImplementedError(f'unsupported generator {generator}')

    # Order of changes
    if order is None or order == 'C':
        pass

    elif order == 'F':
        rand_change = rand_change.reshape(-1).reshape(pChangeP1.shape, order='F')

    else:
        raise NotImplementedError(f'Given order {order} is not implemented')

    # Set up ndarray with simulated changes
    delta = np.zeros(pChangeP1.shape, dtype='int8')

    # rand_change < pChangeP1 => increment 1
    # rand_change < pChangeP1 + pChangeP2 => decrement 1

    delta[rand_change < pChangeP1] = 1
    delta[(rand_change >= pChangeP1) & (rand_change < pChangeP1+pChangeM1)] = -1
    return delta


def ternary(
    rhoP1: np.ndarray,
    rhoM1: np.ndarray,
    alpha: float,
    n: int,
    **kw,
) -> np.ndarray:
    """Simulates ternary embedding given distortion and embedding rate.

    Args:
        rhoP1, rhoM1 (np.ndarray): Distortion tensors for +1 and -1.
        alpha (float): Embedding rate.
        n (int): Cover size (only embeddable elements).
        kw (dict): Additional arguments passed on to _ternary.simulate().
    Returns:
        (np.ndarray): Simulated binary changes in the cover, 0 (keep), 1 or -1 (change).
    """
    (pChangeP1, pChangeM1), lbda = probability(
        rhoP1=rhoP1,
        rhoM1=rhoM1,
        alpha=alpha,
        n=n,
    )
    return simulate(pChangeP1, pChangeM1, **kw)
