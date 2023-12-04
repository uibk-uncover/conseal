"""

Author: Benedikt Lorch, Martin Benes
Affiliation: University of Innsbruck
"""

# costmap
from . import _costmap
from ._costmap import compute_distortion, Implementation

# simulate
from . import _simulate
from ._simulate import simulate_single_channel

# abbreviations of enum
JUNIWARD_ORIGINAL = Implementation.JUNIWARD_ORIGINAL
JUNIWARD_FIX_OFF_BY_ONE = Implementation.JUNIWARD_FIX_OFF_BY_ONE

__all__ = [
    '_costmap',
    '_simulate',
    'compute_distortion',
    'simulate_single_channel',
    'Implementation',
    'JUNIWARD_ORIGINAL',
    'JUNIWARD_FIX_OFF_BY_ONE',
]
