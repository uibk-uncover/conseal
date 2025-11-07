"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Callable, Tuple

from .. import common
from ... import simulate as sim
from ... import tools


def average_payload(
    *,
    ps: Tuple[np.ndarray] = None,
    e: float = None,
    lbda: float,
    rhos: Tuple[np.ndarray] = None,
    q: int = None,
) -> float:
    """"""
    assert (
        (ps is not None and rhos is None or ps is None and rhos is not None)
    ), 'one of ps or rhos must be given'
    assert (
        lbda is not None or ps is not None
    ), 'lbda can be specified only with rhos'
    if ps is None:
        ps = ([
            sim.get_p(rhos[0][c], lbda)
            for c in range(len(rhos[0]))
        ],)

    # Imperfect coding - given embedding efficiency
    if e is not None:
        H = np.sum(ps) * e
    # Perfect coding - upper bound efficiency
    elif q is None or len(ps) == q-1:  # no change is zero cost
        raise NotImplementedError
    else:
        H = np.sum([
            tools.entropy(ps[0][c])
            for c in range(len(ps[0]))
        ])
    return ps, H


def get_objective(
    sender: sim.Sender = sim.Sender.PAYLOAD_LIMITED_SENDER,
    e: float = None,
    q: int = None,
) -> Callable:
    def _pls_objective(*args, **kw):
        return average_payload(*args, e=e, q=q, **kw)

    if sender == sim.Sender.PAYLOAD_LIMITED_SENDER:
        assert e is None, 'e not implemented for DLS'
        return average_payload if e is None else _pls_objective
    if sender == sim.Sender.DISTORTION_LIMITED_SENDER:
        assert e is None, 'e not implemented for DLS'
        raise NotImplementedError
        # return average_distortion


def probability(
    *args,
    e: float = None,
    objective: Callable = None,
    sender: sim.Sender = sim.PAYLOAD_LIMITED_SENDER,
    **kw,
) -> Tuple[Tuple[np.ndarray], float]:
    """Convert binary distortion to binary probability.

    :param args: positional arguments passed to cl.simulate._binary.probability
    :param objective: Objective function to maximize.
        Per-channel entropy by default.
    :type objective: callable
    :param kw: keyword arguments passed to cl.simulate._binary.probability
    :return: tuple ((p_pm1), lmbda), where
        p_pm1 is a list of probabilities of +-1 change in each color channel, and
        lbda is the determined lambda.
    :rtype: tuple
    """
    if objective is None:
        objective = get_objective(e=e, q=2, sender=sender)
        # objective = average_payload

    return sim._binary.probability(*args, objective=objective, **kw)


def simulate(
    ps: Tuple[np.ndarray],
    *,
    seed: int = None,
    stack_axis: int = -1,
    **kw,
) -> np.ndarray:
    """Simulates changes using the given probability maps.

    :param ps: list of probability tensors for +-1 change for each color channel,
        of an arbitrary shape
    :type p_p1: tuple of tuples with a single `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param seed: random seed for embedding simulator
    :type seed: int
    :param stack_axis: axis along which to stack the result, no stacking if None
    :type stack_axis: int
    :return: Simulated binary changes in the cover, 0 (keep), +1 or -1.
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.color.map_channels(cl.hill.compute_cost, x0)
    >>> ps, _ = cl.color.joint.probability(rhos=rhos, alpha=.4, n=x0.size)
    >>> delta = cl.color.joint.simulate(ps=ps, seed=12345)
    """
    #
    seeds = common.seed_per_channel(seed)
    #
    (p_pm1,) = ps
    x1 = [
        sim._binary.simulate(
            ps=(p_pm1[c],),
            seed=seeds[c],
            **kw
        ) for c in range(p_pm1.shape[0])
    ]
    #
    if stack_axis is not None:
        return np.stack(x1, axis=stack_axis)
    else:
        return x1


def binary(
    rhos: Tuple[np.ndarray],
    alpha: float,
    n: int,
    **kw,
) -> np.ndarray:
    """Simulates binary embedding given distortion and embedding rate.

    :param rhos: cost for +-1 changes per channel,
        of an arbitrary shape
    :type rhos: tuple of tuples of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per element
    :type alpha: float
    :param n: cover size/number of cover elements
    :type n: int
    :return: Simulated difference image to be added to the cover, 0 (keep), 1 or -1 (change).
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.color.map_channels(cl.hill.compute_cost, x0)
    >>> delta = cl.color.joint.binary(rhos=rhos, alpha=.4, n=x0.size, seed=12345)
    """
    ps, _ = probability(rhos=rhos, alpha=alpha, n=n)
    return simulate(ps, **kw)
