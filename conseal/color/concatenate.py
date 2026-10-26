"""Simulation over concatenated color channels, with one lambda for all of them.

The channels may differ in shape (e.g. subsampled chroma),
so they are flattened and concatenated for the search of lambda,
and the probabilities are split back into the channels.
The changes are sampled with a seed per channel.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Callable, List, Tuple

from . import _common
from .. import simulate as sim


def _flatten(xs: List[np.ndarray]) -> np.ndarray:
    """Flattens and concatenates the channels."""
    return np.concatenate([np.ravel(x) for x in xs])


def _split(x: np.ndarray, shapes: List[Tuple[int]]) -> List[np.ndarray]:
    """Inverse of _flatten."""
    sizes = [int(np.prod(s)) for s in shapes]
    parts = np.split(x, np.cumsum(sizes)[:-1])
    return [p.reshape(s) for p, s in zip(parts, shapes)]


def probability(
    rhos: Tuple[List[np.ndarray]],
    alpha: float,
    n: int = None,
    **kw,
) -> Tuple[Tuple[List[np.ndarray]], float]:
    """Converts costs of all channels into change probabilities, with one lambda.

    :param rhos: costs of the changes, one list of channels per change,
        channels may differ in shape (e.g. subsampled chroma)
    :type rhos: tuple of lists of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate, in bits per element of all channels
    :type alpha: float
    :param n: number of elements, of all channels by default
    :type n: int
    :param kw: arguments of :func:`conseal.simulate.probability`,
        e.g., objective, solver, or embedding efficiency e
    :return: tuple (ps, lbda) of probabilities,
        one list of channels per change, and λ
    :rtype: tuple

    :Example:

    >>> rhos = cl.color.map_channels(cl.hill.compute_cost_adjusted, x0, stack_axis=None)
    >>> (p_p1, p_m1), lbda = cl.color.concatenate.probability(
    ...     rhos=cl.color.transpose01(rhos),  # costs of +1 and -1, per channel
    ...     alpha=.4)                         # bits per element
    """
    shapes = [np.shape(r) for r in rhos[0]]
    if n is None:
        n = int(np.sum([np.prod(s) for s in shapes]))
    ps, lbda = sim.probability(
        tuple(_flatten(rho) for rho in rhos),
        alpha=alpha,
        n=n,
        **kw,
    )
    return tuple(_split(p, shapes) for p in ps), lbda


def sample(
    ps: Tuple[List[np.ndarray]],
    *,
    seed: int = None,
    stack_axis: int = -1,
    **kw,
) -> np.ndarray | List[np.ndarray]:
    """Samples the changes of each channel, with a seed per channel.

    :param ps: probabilities of the changes, one list of channels per change
    :type ps: tuple of lists of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param seed: random seed, from which the seeds of the channels are derived
    :type seed: int
    :param stack_axis: axis along which to stack the channels, no stacking if None
    :type stack_axis: int
    :param kw: arguments of :func:`conseal.simulate.sample`,
        e.g., values, generator, or order
    :return: simulated changes, 0 for no change
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__ or list

    :Example:

    >>> ps, _ = cl.color.concatenate.probability(rhos=rhos, alpha=.4)
    >>> delta = cl.color.concatenate.sample(ps, seed=12345)
    """
    num_channels = len(ps[0])
    seeds = _common.seed_per_channel(seed, num_channels)
    delta = [
        sim.sample(tuple(p[c] for p in ps), seed=seeds[c], **kw)
        for c in range(num_channels)
    ]
    if stack_axis is not None:
        return np.stack(delta, axis=stack_axis)
    else:
        return delta


def simulate(
    rhos: Tuple[List[np.ndarray]],
    alpha: float,
    n: int = None,
    seed: int = None,
    *,
    stack_axis: int = -1,
    values: Tuple[int] = None,
    generator: str | Callable = None,
    order: str = 'C',
    **kw,
) -> np.ndarray | List[np.ndarray]:
    """Simulates embedding changes of all channels, given costs and embedding rate.

    Combines :func:`probability` and :func:`sample`.

    :param rhos: costs of the changes, one list of channels per change
    :type rhos: tuple of lists of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate, in bits per element of all channels
    :type alpha: float
    :param n: number of elements, of all channels by default
    :type n: int
    :param seed: random seed, from which the seeds of the channels are derived
    :type seed: int
    :param stack_axis: axis along which to stack the channels, no stacking if None
    :type stack_axis: int
    :param values: values of the changes, see :func:`conseal.simulate.sample`
    :type values: tuple
    :param generator: random number generator, see :func:`conseal.simulate.sample`
    :type generator: str or callable
    :param order: order of changes, see :func:`conseal.simulate.sample`
    :type order: str
    :param kw: arguments of :func:`probability`,
        e.g., objective, solver, or embedding efficiency e
    :return: simulated changes, 0 for no change
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__ or list

    :Example:

    >>> rhos = cl.color.map_channels(cl.hill.compute_cost_adjusted, x0, stack_axis=None)
    >>> delta = cl.color.concatenate.simulate(
    ...     rhos=cl.color.transpose01(rhos),  # costs of +1 and -1, per channel
    ...     alpha=.4,                         # bits per element
    ...     seed=12345)                       # seed
    """
    ps, _ = probability(rhos, alpha, n, **kw)
    return sample(
        ps,
        seed=seed,
        stack_axis=stack_axis,
        values=values,
        generator=generator,
        order=order,
    )


__all__ = ['probability', 'sample', 'simulate']
