"""Simulator of steganographic embedding, based on the Gibbs distribution.

The embedding probabilities follow p ~ exp(-λ ρ),
where λ is found by search(), s.t., the objective f(λ) meets the target.

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Tuple, Callable
import warnings

from ._common import Sender, get_probability
from ._objective import (
    Objective,
    distortion,
    d_distortion,
    distortion_limits,
    entropy,
    d_entropy,
    entropy_limits,
)
from ._solver import LambdaSolver
from . import _common
from . import _objective
from . import _solver

PAYLOAD_LIMITED_SENDER = Sender.PAYLOAD_LIMITED_SENDER
DISTORTION_LIMITED_SENDER = Sender.DISTORTION_LIMITED_SENDER
PLS = PAYLOAD_LIMITED_SENDER
DLS = DISTORTION_LIMITED_SENDER
SOLVER_BSEARCH_DDE = LambdaSolver.SOLVER_BSEARCH_DDE
SOLVER_BSEARCH = LambdaSolver.SOLVER_BSEARCH
SOLVER_NEWTON = LambdaSolver.SOLVER_NEWTON
SOLVER_BRENT = LambdaSolver.SOLVER_BRENT


def search(
    target: float,
    rhos: Tuple[np.ndarray],
    *,
    solver: LambdaSolver = LambdaSolver.SOLVER_NEWTON,
    objective: Sender | Objective | Callable = Sender.PAYLOAD_LIMITED_SENDER,
    lbda0: float | Tuple[int] = None,
    tol: float = None,  # solver default if None
    max_iter: int = None,  # solver default if None
    n: int = None,
    q: int = None,
    e: float = None,
    **kw,
) -> float:
    """Search for λ, s.t., objective f(λ) = target.

    Targets outside of [f(inf), f(0+)] have no solution.
    For f(0+) <= target, returns 0, for target <= f(inf), returns inf.
    DDE solver is exempt from this, to keep compatibility.

    :param target: desired value of the objective
    :type target: float
    :param rhos: embedding costs
    :type rhos: tuple of `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param solver: solver to use
    :type solver: :class:`LambdaSolver`
    :param objective: sender to dispatch the objective by,
        custom objective, or
        callable as objective without derivative and limits
    :type objective: :class:`Sender`, :class:`Objective` or callable
    :param e: embedding efficiency, only with sender,
        optimal coding by default
    :type e: float
    :return: estimated λ value
    :rtype: float
    """
    objective = Objective.of(objective, e=e)
    if q is None:
        q = len(rhos) + 1
    # pass only what is given, solvers have their own defaults
    kw.update({
        k: v
        for k, v in dict(tol=tol, max_iter=max_iter, n=n).items()
        if v is not None
    })
    #
    # targets outside of [f(inf), f(0+)] have no solution
    limits = objective.limits(rhos, q=q)
    if limits is not None and solver != LambdaSolver.SOLVER_BSEARCH_DDE:
        f_0, f_inf = limits
        if target >= f_0:
            return 0.
        if target <= f_inf:
            warnings.warn("target unreachable, lambda is infinite", RuntimeWarning)
            return np.inf
    #
    if solver.expsearch is not None:
        lbda1, _ = solver.expsearch(
            target=target,
            rhos=rhos,
            objective=objective,
            lbda0=lbda0,
            q=q,
        )
    else:
        lbda1 = lbda0
    #
    lbda = solver.solve(
        target=target,
        rhos=rhos,
        objective=objective,
        lbda1=lbda1,
        q=q,
        **kw,
    )
    return lbda


# simulator, uses search()
from ._simulator import probability, sample, simulate  # noqa: E402
from . import _simulator  # noqa: E402

# deprecated interface of the legacy simulator, imported last
from ._legacy import (  # noqa: E402
    _binary,
    _ternary,
    binary,
    ternary,
    get_p,
    average_payload,
    average_distortion,
)


__all__ = [
    'Sender',
    'Objective',
    'LambdaSolver',
    'search',
    'get_probability',
    'probability',
    'sample',
    'simulate',
    #
    'distortion',
    'd_distortion',
    'distortion_limits',
    'entropy',
    'd_entropy',
    'entropy_limits',
    #
    'PAYLOAD_LIMITED_SENDER',
    'DISTORTION_LIMITED_SENDER',
    'PLS',
    'DLS',
    'SOLVER_BSEARCH_DDE',
    'SOLVER_BSEARCH',
    'SOLVER_NEWTON',
    'SOLVER_BRENT',
    # deprecated
    '_binary',
    '_ternary',
    'binary',
    'ternary',
    'get_p',
    'average_payload',
    'average_distortion',
]
