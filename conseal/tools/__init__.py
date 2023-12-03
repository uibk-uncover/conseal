"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

# globals
from .common import entropy, inv_entropy, _entropy, EPS
from .dct import nzAC

# modules
from . import dct
from . import jpeg
from . import spatial


__all__ = [
    'entropy',
    'inv_entropy',
    '_entropy',
    'EPS',
    'nzAC',
    'dct',
    'jpeg',
    'spatial',
]
