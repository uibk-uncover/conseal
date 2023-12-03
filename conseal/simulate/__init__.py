"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Tuple, Union

from ._defs import get_p
from . import _ternary
from ._ternary import ternary

__all__ = [
    '_costmap',
    '_simulate',
    'compute_distortion',
    'compute_cost',
    'simulate_single_channel',
]


def simulate(
    rho: Union[np.ndarray, Tuple[np.ndarray]],
    alpha: float,
    n: int,
    color_strategy: str = None,
    gamma: float = .25,
    seed: int = None,
    **kw,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    # multiple rhos given
    if not isinstance(rho, np.ndarray):
        q = len(rho) + 1
    # single rho given
    else:
        q = 2

    # # color strategy
    # if color_strategy is None:
    #     pass
    # if color_strategy.lower() in {'ccfr', 'joint', 'arb', 'repartition'}:
    #     pass
    #     # use gamma
    # elif color_strategy.lower() in {'r', 'red'}:
    #     pass
    # elif color_strategy.lower() in {'g', 'green'}:
    #     pass
    # elif color_strategy.lower() in {'b', 'blue'}:
    #     pass
    # elif color_strategy.lower() in {'y', 'lumo', 'luminance', 'gray'}:
    #     pass
    # elif color_strategy.lower() in {'cb'}:
    #     pass
    # elif color_strategy.lower() in {'cr'}:
    #     pass
    # else:
    #     raise NotImplementedError(f'unknown color strategy {color_strategy}')


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
