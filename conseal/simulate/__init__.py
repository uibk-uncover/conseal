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

from ._optim import get_p, average_payload, average_distortion, Sender

PAYLOAD_LIMITED_SENDER = Sender.PAYLOAD_LIMITED_SENDER
DISTORTION_LIMITED_SENDER = Sender.DISTORTION_LIMITED_SENDER
PLS = PAYLOAD_LIMITED_SENDER
DLS = DISTORTION_LIMITED_SENDER


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
    '_optim',
    '_ternary',
    'ternary',
    '_binary',
    'binary',
    'get_p',
    'simulate',
    'average_payload',
]
