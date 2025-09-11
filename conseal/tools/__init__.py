"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

# globals
from .common import entropy, inv_entropy, _entropy, EPS32, EPS64, EPS, matlab_round
from ._backend import Backend, set_backend, get_backend
from .dct import nzAC, AC, jpegio_to_jpeglib, jpeglib_to_jpegio
from .decompress import decompress_channel
from .permute import blocks, iterate_ac, iterate, password_to_seed

# modules
from . import _backend
from . import dct
from . import spatial
from . import lrt

#
BACKEND_PYTHON = Backend.BACKEND_PYTHON
BACKEND_RUST = Backend.BACKEND_RUST
#
ATTACKER_INDIFFERENT = lrt.Attacker.ATTACKER_INDIFFERENT
ATTACKER_OMNISCIENT = lrt.Attacker.ATTACKER_OMNISCIENT

__all__ = [
    'entropy',
    'inv_entropy',
    '_entropy',
    'EPS',
    'nzAC',
    'AC',
    'dct',
    'spatial',
]
