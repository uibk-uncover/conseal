"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

# simulate
from . import _simulate
from ._simulate import simulate_single_channel

__all__ = [
    '_simulate',
    'simulate_single_channel',
]
