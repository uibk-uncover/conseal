"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
from typing import Tuple

from . import _binary
from . import _ternary
from ._binary import binary
from ._ternary import ternary

from ._defs import Sender
from .optim import get_p, average_payload
from .optim_new import get_objective, get_d_objective, get_probability, LambdaOptimizer
from .optim_new import (
    expected_distortion,
    d_expected_distortion,
    entropy_with_logit, entropy,
    d_entropy_with_logit,
    d_entropy
)
# from ._optim_binary import (
#     expected_distortion,
#     d_expected_distortion,
#     # entropy_with_logit,
#     # entropy,
#     # d_entropy,
#     # binary_search,
#     # newton,
#     # polynomial_proxy,
#     # taylor_inverse,
#     # compare_methods,
# )

PAYLOAD_LIMITED_SENDER = Sender.PAYLOAD_LIMITED_SENDER
DISTORTION_LIMITED_SENDER = Sender.DISTORTION_LIMITED_SENDER
PAYLOAD_LIMITED_SENDER_DDE = Sender.PAYLOAD_LIMITED_SENDER_DDE
PLS = PAYLOAD_LIMITED_SENDER
PLS_DDE = PAYLOAD_LIMITED_SENDER_DDE
DLS = DISTORTION_LIMITED_SENDER
BINARY_SEARCH_DDE = LambdaOptimizer.BINARY_SEARCH_DDE
BINARY_SEARCH = LambdaOptimizer.BINARY_SEARCH
NEWTON = LambdaOptimizer.NEWTON
POLYNOMIAL_PROXY = LambdaOptimizer.POLYNOMIAL_PROXY
TAYLOR_INVERSE = LambdaOptimizer.TAYLOR_INVERSE
# TAYLOR_NEWTON = LambdaOptimizer.TAYLOR_NEWTON
# BINARY_SEARCH_NEWTON = LambdaOptimizer.BINARY_SEARCH_NEWTON


def simulate(
    rhos: Tuple[np.ndarray],
    alpha: float,
    n: int,
    seed: int = None,
    q: int = None,
    **kw,
) -> Tuple[np.ndarray]:
    """

    :param rhos: either
        a distortion tensor for +-1 change, or
        a tuple with tensors for +1 and -1 change
    :type rho: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param alpha: embedding rate
    :type alpha: float
    :param n: Cover size.
    :type n: int
    :param seed: random seed for embedding simulator
    :type seed: int
    :return:
    :rtype: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__

    :Example:

    >>> im_dct.Y += cl.simulate.ternary(
    ...     rhos=rhos,        # costs of change
    ...     alpha=0.4,        # alpha
    ...     n=im_dct.Y.size,  # cover size
    ...     seed=12345)       # seed
    """
    # derive q if not given
    if q is None:
        q = len(rhos) + 1

    # ternary
    if q == 3:
        return _ternary.ternary(
            rhos=rhos,
            alpha=alpha,
            n=n,
            seed=seed,
            **kw,
        )
    elif q == 2:
        return _binary.binary(
            rhos=rhos,
            alpha=alpha,
            n=n,
            seed=seed,
            **kw,
        )
    # other
    else:
        raise NotImplementedError(f'{q=} not implemented')


__all__ = [
    'optim',
    '_ternary',
    'ternary',
    '_binary',
    'binary',
    'get_p',
    'simulate',
    'average_payload',
    'get_objective',
    'get_d_objective',
    'get_probability',
    'LambdaOptimizer',
    #
    'expected_distortion',
    'd_expected_distortion',
    'entropy_with_logit',
    'entropy',
    'd_entropy_with_logit',
    'd_entropy',
]
