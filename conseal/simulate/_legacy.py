"""Deprecated interface of the legacy simulator.

Mimics the former interface of conseal.simulate with a deprecation warning,
delegating to the replacement where available, and to conseal.simulate_old otherwise.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import functools
import types
from typing import Callable
import warnings

from .. import simulate_old
from ._simulator import sample


def _deprecated(func: Callable, name: str, replacement: str = None) -> Callable:
    """Wraps legacy function with a deprecation warning."""
    if replacement is None:
        replacement = f'conseal.simulate_old.{name} until its replacement'

    @functools.wraps(func)
    def wrapper(*args, **kw):
        warnings.warn(
            f'conseal.simulate.{name} is deprecated, use {replacement}',
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kw)
    return wrapper


@functools.wraps(simulate_old._binary.simulate)
def _sample_binary(ps, generator=None, order='C', seed=None):
    return sample(ps[:1], generator=generator, order=order, seed=seed)


@functools.wraps(simulate_old._ternary.simulate)
def _sample_ternary(ps, generator=None, order='C', seed=None):
    return sample(ps[:2], generator=generator, order=order, seed=seed)


binary = _deprecated(simulate_old.binary, 'binary')
ternary = _deprecated(simulate_old.ternary, 'ternary')
get_p = _deprecated(simulate_old.get_p, 'get_p')
average_payload = _deprecated(simulate_old.average_payload, 'average_payload')
average_distortion = _deprecated(simulate_old.average_distortion, 'average_distortion')
calc_lambda = _deprecated(simulate_old.calc_lambda, 'calc_lambda', 'conseal.simulate.search')

_binary = types.SimpleNamespace(
    probability=_deprecated(simulate_old._binary.probability, '_binary.probability'),
    simulate=_deprecated(_sample_binary, '_binary.simulate', 'conseal.simulate.sample'),
    binary=_deprecated(simulate_old._binary.binary, '_binary.binary'),
)
_ternary = types.SimpleNamespace(
    probability=_deprecated(simulate_old._ternary.probability, '_ternary.probability'),
    simulate=_deprecated(_sample_ternary, '_ternary.simulate', 'conseal.simulate.sample'),
    ternary=_deprecated(simulate_old._ternary.ternary, '_ternary.ternary'),
)


__all__ = [
    '_binary',
    '_ternary',
    'binary',
    'ternary',
    'get_p',
    'average_payload',
    'average_distortion',
    'calc_lambda',
]
