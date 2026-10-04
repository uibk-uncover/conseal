"""Simulator of embedding changes.

Converts costs into change probabilities,
and samples the changes from the probabilities.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Tuple, Callable

from ._common import Sender, get_probability
from ._objective import Objective
from ._solver import LambdaSolver


def probability(
    rhos: np.ndarray | Tuple[np.ndarray],
    alpha: float,
    n: int = None,
    *,
    objective: Sender | Objective | Callable = Sender.PAYLOAD_LIMITED_SENDER,
    solver: LambdaSolver = LambdaSolver.SOLVER_BSEARCH_DDE,
    e: float = None,
    q: int = None,
    **kw,
) -> Tuple[np.ndarray, float]:
    """Converts costs into change probabilities.

    Searches λ, s.t., the objective meets alpha * n,
    and returns the probabilities p ~ exp(-λ ρ).
    For payload-limited sender, the target is rounded to whole bits, as in DDE.

    :param rhos: costs of the changes, one tensor per change
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate,
        in bits per element for payload-limited sender, or
        in cost per element for distortion-limited sender
    :type alpha: float
    :param n: cover size, size of the costs by default
    :type n: int
    :param objective: sender, objective or callable, see :func:`search`
    :type objective: :class:`Sender`, :class:`Objective` or callable
    :param solver: solver of the λ search,
        DDE-compatible binary search by default
    :type solver: :class:`LambdaSolver`
    :param e: embedding efficiency, optimal coding by default
    :type e: float
    :param q: number of states, len(rhos) + 1 by default,
        i.e., no change is implicit with zero cost
    :type q: int
    :return: tuple (ps, lbda) of probabilities, one tensor per change, and λ
    :rtype: tuple

    :Example:

    >>> (p_p1, p_m1), lbda = cl.simulate.probability(
    ...     rhos=(rho_p1, rho_m1),  # costs of +1 and -1
    ...     alpha=.4)               # bits per element
    """
    from . import search  # defined in the package, imports this module
    rhos = np.array(rhos)
    if n is None:
        n = rhos[0].size
    if q is None:
        q = len(rhos) + 1
    objective = Objective.of(objective, e=e)
    # target
    target = alpha * n
    if objective.sender == Sender.PAYLOAD_LIMITED_SENDER:
        target = int(np.round(target))  # message length in bits
    # λ search
    lbda = search(
        target=target,
        rhos=rhos,
        objective=objective,
        solver=solver,
        n=n,
        q=q,
        **kw,
    )
    ps = get_probability(rhos=rhos, lbda=lbda, add_zero=len(rhos) == q-1)
    return ps, lbda


def sample(
    ps: np.ndarray | Tuple[np.ndarray],
    *,
    values: Tuple[int] = None,
    generator: str | Callable = None,
    order: str = 'C',
    seed: int = None,
) -> np.ndarray:
    """Samples the changes from the change probabilities.

    Draws one uniform number per element,
    and selects the change, whose cumulative probability interval contains it.
    Without any, the element is not changed.

    :param ps: probabilities of the changes, one tensor per change,
        of an arbitrary shape
    :type ps: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param values: values of the changes,
        (1,) for one change and (1, -1) for two changes by default
    :type values: tuple
    :param generator: random number generator,
        None (numpy default), 'MT19937' (used by Matlab),
        or callable returning uniform numbers of given shape
    :type generator: str or callable
    :param order: order of changes,
        'C' (C-order, row-major) or 'F' (F-order, column-major)
    :type order: str
    :param seed: random seed
    :type seed: int
    :return: simulated changes, 0 for no change
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y += cl.simulate.sample(
    ...     ps=(p_p1, p_m1),  # probabilities of +1 and -1
    ...     seed=12345)       # seed
    """
    ps = np.array(ps)
    shape = ps.shape[1:]
    # values of changes
    if values is None:
        if len(ps) == 1:
            values = (1,)
        elif len(ps) == 2:
            values = (1, -1)
        else:
            raise ValueError(f'values must be given for {len(ps)} changes')
    if len(values) != len(ps):
        raise ValueError(f'expected {len(ps)} values, got {len(values)}')

    # select random number generator
    if generator is None:  # numpy default generator
        rng = np.random.default_rng(seed=seed)
        rand_change = rng.random(shape)
    elif generator == 'MT19937':  # Matlab generator
        prng = np.random.RandomState(seed)
        rand_change = prng.random_sample(shape)
    elif callable(generator):
        rand_change = generator(shape)
    else:
        raise NotImplementedError(f'unsupported generator {generator}')

    # order of changes
    if order is None or order == 'C':
        pass
    elif order == 'F':
        rand_change = rand_change.reshape(-1).reshape(shape, order='F')
    else:
        raise NotImplementedError(f'Given order {order} is not implemented')

    # index of the change, len(ps) for no change
    index = np.sum(rand_change[None] >= np.cumsum(ps, axis=0), axis=0)
    return np.append(np.array(values, dtype='int8'), 0)[index]


def simulate(
    rhos: np.ndarray | Tuple[np.ndarray],
    alpha: float,
    n: int = None,
    seed: int = None,
    *,
    values: Tuple[int] = None,
    generator: str | Callable = None,
    order: str = 'C',
    **kw,
) -> np.ndarray:
    """Simulates embedding changes, given costs and embedding rate.

    Combines :func:`probability` and :func:`sample`.

    :param rhos: costs of the changes, one tensor per change
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate, see :func:`probability`
    :type alpha: float
    :param n: cover size, size of the costs by default
    :type n: int
    :param seed: random seed
    :type seed: int
    :param values: values of the changes, see :func:`sample`
    :type values: tuple
    :param generator: random number generator, see :func:`sample`
    :type generator: str or callable
    :param order: order of changes, see :func:`sample`
    :type order: str
    :param kw: arguments of :func:`probability`,
        e.g., objective, solver, or embedding efficiency e
    :return: simulated changes, 0 for no change
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y += cl.simulate.simulate(
    ...     rhos=(rho_p1, rho_m1),  # costs of +1 and -1
    ...     alpha=.4,               # bits per element
    ...     seed=12345)             # seed
    """
    ps, _ = probability(rhos=rhos, alpha=alpha, n=n, **kw)
    return sample(ps, values=values, generator=generator, order=order, seed=seed)


__all__ = ['probability', 'sample', 'simulate']
