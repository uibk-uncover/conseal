"""Joint changes of the color channels of a pixel.

A pixel of C channels is changed jointly, by one of the change patterns.
Each channel takes either no change, or one of the changes,
+-1 (symmetric) or +1 and -1 (asymmetric).
This gives 2**C or 3**C joint states, including the no change.

The marginal costs or probabilities of the channels expand
into the joint ones of the patterns, and the joint ones marginalize back.
The no change is implicit, as in :mod:`conseal.simulate`,
so the joint tensors have 2**C - 1 or 3**C - 1 patterns.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Tuple


def _digits(num_channels: int, num_changes: int) -> np.ndarray:
    """Change of each channel in each pattern, 0 for no change,
    i for the i-th change, the first channel varying fastest.

    :return: indices of shape [patterns, channels], without the no change
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    """
    base = num_changes + 1
    index = np.arange(1, base**num_channels)
    return (index[:, None] // base**np.arange(num_channels)) % base


def _num_channels(num_patterns: int, num_changes: int) -> int:
    """Number of channels, from the number of patterns without the no change."""
    base = num_changes + 1
    num_channels = int(np.round(np.log(num_patterns + 1) / np.log(base)))
    if base**num_channels != num_patterns + 1:
        raise ValueError(f'{num_patterns} patterns are not {base}**C - 1 for any C')
    return num_channels


def change_patterns(
    num_channels: int = 3,
    *,
    symmetric: bool = False,
) -> np.ndarray:
    """Gives the change patterns of a pixel.

    The pattern translates the joint state into the change of each channel.
    The first channel varies fastest.

    :param num_channels: number of channels C, 3 by default
    :type num_channels: int
    :param symmetric: whether the change is +-1 with one cost (2**C states),
        or +1 and -1 with separate costs (3**C states), asymmetric by default
    :type symmetric: bool
    :return: changes of shape [patterns, channels],
        without the no change, i.e., 2**C - 1 or 3**C - 1 patterns
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> cl.color.joint.change_patterns(2)
    array([[ 1,  0],
           [-1,  0],
           [ 0,  1],
           [ 1,  1],
           [-1,  1],
           [ 0, -1],
           [ 1, -1],
           [-1, -1]], dtype=int8)
    """
    values = np.array([0, 1] if symmetric else [0, 1, -1], dtype='int8')
    return values[_digits(num_channels, len(values) - 1)]


def from_marginal(
    rhos: Tuple[np.ndarray] = None,
    *,
    ps: Tuple[np.ndarray] = None,
    channel_axis: int = -1,
) -> np.ndarray:
    """Expands marginal costs or probabilities of the channels into joint ones.

    The joint cost is the sum of the costs of the channels.
    The joint probability is the product of the probabilities of the channels,
    i.e., the channels change independently.

    :param rhos: costs of the changes, one tensor per change,
        (rho_pm1,) for symmetric or (rho_p1, rho_m1) for asymmetric,
        with the channels along channel_axis
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps: probabilities of the changes, instead of rhos
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param channel_axis: axis of the channels
    :type channel_axis: int
    :return: joint costs or probabilities of shape [patterns, ...],
        in the order of :func:`change_patterns`
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> rhos = cl.hill.compute_cost_adjusted(x0)  # x0 of shape [H, W, 3]
    >>> rhos_joint = cl.color.joint.from_marginal(rhos)  # [26, H, W]
    """
    if (rhos is None) == (ps is None):
        raise ValueError('exactly one of rhos or ps must be given')
    xs = np.moveaxis(np.array(rhos if ps is None else ps), channel_axis + (channel_axis >= 0), 1)
    num_channels = xs.shape[1]
    # prepend the no change
    if ps is None:
        x0 = np.zeros_like(xs[:1])
    else:
        x0 = 1 - np.sum(xs, axis=0, keepdims=True)
    xs = np.concatenate([x0, xs])
    # select the change of each channel, [patterns, channels, ...]
    digits = _digits(num_channels, len(xs) - 1)
    xs = xs[digits, np.arange(num_channels)]
    if ps is None:
        return np.sum(xs, axis=1)
    else:
        return np.prod(xs, axis=1)


def to_marginal(
    rhos: np.ndarray = None,
    *,
    ps: np.ndarray = None,
    symmetric: bool = False,
    channel_axis: int = -1,
) -> Tuple[np.ndarray]:
    """Marginalizes joint costs or probabilities into those of the channels.

    The marginal probability of a change of a channel
    sums the joint probabilities of the patterns with that change.
    The marginal cost of a change of a channel is the joint cost
    of the pattern, which changes only that channel,
    i.e., the inverse of :func:`from_marginal` for additive costs.

    :param rhos: joint costs of shape [patterns, ...],
        in the order of :func:`change_patterns`
    :type rhos: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ps: joint probabilities, instead of rhos
    :type ps: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param symmetric: whether the patterns are symmetric, asymmetric by default
    :type symmetric: bool
    :param channel_axis: axis along which to stack the channels
    :type channel_axis: int
    :return: marginal costs or probabilities, one tensor per change,
        (x_pm1,) for symmetric or (x_p1, x_m1) for asymmetric
    :rtype: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> ps, _ = cl.simulate.probability(rhos_joint, alpha=.4, n=x0.size)
    >>> p_p1, p_m1 = cl.color.joint.to_marginal(ps=ps)  # [H, W, 3]
    """
    if (rhos is None) == (ps is None):
        raise ValueError('exactly one of rhos or ps must be given')
    xs = np.array(rhos if ps is None else ps)
    num_changes = 1 if symmetric else 2
    num_channels = _num_channels(len(xs), num_changes)
    digits = _digits(num_channels, num_changes)
    #
    marginals = []
    for i in range(1, num_changes + 1):
        if ps is None:
            # pattern changing only channel c, by the i-th change
            index = i * (num_changes + 1)**np.arange(num_channels) - 1
            x = [xs[index[c]] for c in range(num_channels)]
        else:
            x = [np.sum(xs[digits[:, c] == i], axis=0) for c in range(num_channels)]
        marginals.append(np.stack(x, axis=channel_axis))
    return tuple(marginals)


__all__ = ['change_patterns', 'from_marginal', 'to_marginal']
