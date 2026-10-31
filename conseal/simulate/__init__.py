"""Joint call of the simulator, and the deprecated interface of the legacy simulator.

conseal.simulate() is conseal.simulator.simulate(),
which converts costs into probabilities and samples the changes.
The simulator itself is in conseal.simulator.

The other names are the interface of the former simulator,
kept for backward compatibility, with a deprecation warning.
Once they are dropped, conseal.simulate becomes a plain function.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import sys
import types

from .. import simulator
from ..simulator import (
    Sender,
    PAYLOAD_LIMITED_SENDER,
    DISTORTION_LIMITED_SENDER,
    PLS,
    DLS,
)
from . import _legacy
from ._legacy import (
    _binary,
    _ternary,
    binary,
    ternary,
    get_p,
    average_payload,
    average_distortion,
    calc_lambda,
    simulate,
)


class _CallableModule(types.ModuleType):
    """Module, which can be called as conseal.simulator.simulate().

    Keeps conseal.simulate a module, so that the legacy names,
    conseal.simulate.ternary or from conseal.simulate import ternary, still work.
    """

    def __call__(self, *args, **kw):
        return simulator.simulate(*args, **kw)


_CallableModule.__call__.__doc__ = simulator.simulate.__doc__
sys.modules[__name__].__class__ = _CallableModule


__all__ = [
    'Sender',
    'PAYLOAD_LIMITED_SENDER',
    'DISTORTION_LIMITED_SENDER',
    'PLS',
    'DLS',
    # deprecated
    '_binary',
    '_ternary',
    'binary',
    'ternary',
    'get_p',
    'average_payload',
    'average_distortion',
    'calc_lambda',
    'simulate',
]
