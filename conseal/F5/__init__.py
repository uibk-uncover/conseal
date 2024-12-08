"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

#
from ._simulate import simulate_single_channel

#
from . import _simulate

__all__ = [
    '_simulate',
    'simulate_single_channel',
]
