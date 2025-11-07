"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Callable, Tuple

from ... import simulate as sim
from ... import tools


def average_payload(
    *,
    ps: Tuple[Tuple[np.ndarray]] = None,
    e: float = None,
    lbda: float = None,
    rhos: Tuple[Tuple[np.ndarray]] = None,
) -> float:
    assert (
        (ps is not None and rhos is None) or (ps is None and rhos is not None)
    ), 'exactly one of ps or rhos must be given'
    assert (
        lbda is not None or ps is not None
    ), 'lambda can be specified only with rhos'

    if ps is None:
        p_p1 = [
            sim.get_p(lbda, rhos[0][ch], rhos[1][ch])
            for ch in range(len(rhos[0]))
        ]
        p_m1 = [
            sim.get_p(lbda, rhos[1][ch], rhos[0][ch])
            for ch in range(len(rhos[0]))
        ]
        ps = (p_p1, p_m1)

    # Imperfect coding - given embedding efficiency
    if e is not None:
        H = np.sum(ps) * e

    # Perfect coding - upper bound efficiency
    # elif len(ps) == 2:
    else:
        H = np.sum([
            tools.entropy(p_p1[ch], p_m1[ch])
            for ch in range(len(p_p1))
        ])

    # Compute ternary entropy
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
    :return: tuple ((p_p1, p_m1), lmbda), where
        p_p1 is a tuple of probabilities of embedding +1 per channel
        p_m1 is a tuple of probabilities of embedding -1 per channel
        lbda is the determined lambda.
    :rtype: tuple
    """
    if objective is None:
        objective = get_objective(e=e, q=3, sender=sender)
        # objective = average_payload

    return sim._ternary.probability(*args, objective=objective, **kw)


def simulate(
    ps: Tuple[Tuple[np.ndarray]],
    *,
    seed: int = None,
    stack_axis: int = -1,
    **kw,
) -> np.ndarray:
    """Simulates changes using the given probability maps.

    :param ps: list of probability tensors for +1 and -1 changes for each color channel,
        of an arbitrary shape
    :type p_p1: tuple of tuples with a two `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param seed: random seed for embedding simulator
    :type seed: int
    :param stack_axis: axis along which to stack the result, no stacking if None
    :type stack_axis: int
    :return: Simulated ternary changes in the cover, 0 (keep), +1, and -1.
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    >>> rhos = cl.color.map_channels(cl.hill.compute_cost_adjusted, x0, stack_channels=None)
    >>> rhos = cl.color.transpose01(rhos)
    >>> ps, _ = cl.color.joint.probability(rhos=rhos, alpha=.4, n=x0.size)
    >>> delta = cl.color.joint.simulate(ps=ps, seed=12345)
    """
    assert len(ps) == 2
    assert len(ps[0]) == len(ps[1])
    C = len(ps[0])
    # get seeds
    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=0, high=2**31, dtype=np.int32, size=C)
    #
    delta = [
        sim._ternary.simulate(
            [ps[i][ch] for i in range(2)],
            seed=seeds[ch],
            **kw,
        ) for ch in range(C)
    ]
    if stack_axis is not None:
        return np.stack(delta, axis=stack_axis)
    else:
        return delta


def ternary(
    rhos: Tuple[np.ndarray],
    alpha: float,
    n: int,
    *,
    e: float = None,
    objective: Callable = None,
    sender: sim.Sender = sim.PAYLOAD_LIMITED_SENDER,
    **kw,
) -> np.ndarray:
    """Simulates ternary embedding given distortion and embedding rate.

    :param rhos: cost for +-1 changes per channel,
        of an arbitrary shape
    :type rhos: tuple of tuples of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per element
    :type alpha: float
    :param n: cover size/number of cover elements
    :type n: int
    :return: Simulated difference image to be added to the cover, 0 (keep), 1 and -1 (change).
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.color.map_channels(cl.hill.compute_cost, x0)
    >>> delta = cl.color.joint.ternary(rhos=rhos, alpha=.4, n=x0.size, seed=12345)
    """
    ps, _ = probability(
        rhos=rhos,
        alpha=alpha,
        n=n,
        e=e,
        objective=objective,
        sender=sender,
    )
    return simulate(ps=ps, **kw)
