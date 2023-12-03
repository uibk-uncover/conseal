"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import typing

from ._defs import get_p
from . import _ternary
from ._ternary import ternary


def simulate(
    rho: typing.Union[np.ndarray, typing.Tuple[np.ndarray]],
    alpha: float,
    n: int,
    seed: int = None,
    **kw,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray]]:
    # multiple rhos given
    if not isinstance(rho, np.ndarray):
        q = len(rho) + 1
    # single rho given
    else:
        q = 2

    # ternary
    if q == 3:
        return _ternary.ternary(
            rho[0], rho[1],
            alpha=alpha,
            n=n,
            seed=seed,
            **kw,
        )
    # other
    else:
        raise NotImplementedError(f'{q=} not implemented')


def average_payload(
    lbda: float,
    rhoPM1: np.ndarray = None,
    rhoP1: np.ndarray = None,
    rhoM1: np.ndarray = None,
    pPM1: np.ndarray = None,
    pP1: np.ndarray = None,
    pM1: np.ndarray = None,
    q: int = 3,
) -> float:
    if q == 3:
        return _ternary.average_payload(
            lbda=lbda,
            rhoP1=rhoP1,
            rhoM1=rhoM1,
            pP1=pP1,
            pM1=pM1,
        )
    else:
        raise NotImplementedError(f'not implemented {q}ary probability')


__all__ = [
    '_defs',
    '_ternary',
    'ternary',
    'get_p',
    'simulate',
    'average_payload',
]
