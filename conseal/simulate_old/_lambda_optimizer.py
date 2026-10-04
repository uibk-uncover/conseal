"""Legacy dispatch of λ optimizers.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import enum

from ..simulate._solver import (
    binary_search,
    binary_search_dde,
    newton,
)


class LambdaOptimizer(enum.Enum):
    """Type of lambda optimizer."""

    BINARY_SEARCH_DDE = enum.auto()
    """Binary search, compatible with DDE's Matlab."""
    BINARY_SEARCH = enum.auto()
    """Binary search."""
    NEWTON = enum.auto()
    """Newton method."""

    def __call__(self, *args, **kw):
        if self == LambdaOptimizer.BINARY_SEARCH_DDE:
            # print('using binary search as optimizer')
            return binary_search_dde(*args, **kw)
        elif self == LambdaOptimizer.BINARY_SEARCH:
            # print('using binary search as optimizer')
            return binary_search(*args, **kw)
        elif self == LambdaOptimizer.NEWTON:
            # print('using newton as optimizer')
            # default initial value obtained via initial lambda search
            if 'lbda0' not in kw:
                (lbda0, _), _ = estimate_lambda_search(*args, **kw)
                kw['lbda0'] = (lbda0 * .75,)
            # apply newton
            return newton(*args, **kw)
        else:
            raise NotImplementedError(f"No implementation for optimizer {self}")
