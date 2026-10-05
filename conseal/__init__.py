
#
from . import coding
from . import color
from . import simulate
from . import simulate_old
from . import tools

# spatial
from . import hill
from . import hugo
from . import lsb  # can be also used for JPEG
from . import mipod
from . import suniward
from . import sunigard
from . import wow
from . import ws

# JPEG
from . import ebs
from . import F5
from . import jmipod
from . import juniward
from . import nsF5
from . import uerd

# backend to choose
from .tools import get_backend, set_backend


# abbreviations of enum
JUNIWARD_ORIGINAL = juniward.Implementation.JUNIWARD_ORIGINAL
JUNIWARD_FIX_OFF_BY_ONE = juniward.Implementation.JUNIWARD_FIX_OFF_BY_ONE
EBS_ORIGINAL = ebs.Implementation.EBS_ORIGINAL
EBS_FIX_WET = ebs.Implementation.EBS_FIX_WET
LSB_REPLACEMENT = lsb.Change.LSB_REPLACEMENT
LSB_MATCHING = lsb.Change.LSB_MATCHING
Location = lsb.Location
LOCATION_PERMUTED = lsb.Location.LOCATION_PERMUTED
LOCATION_SEQUENTIAL = lsb.Location.LOCATION_SEQUENTIAL
LOCATION_SELECTED = lsb.Location.LOCATION_SELECTED
MiPOD_ORIGINAL = mipod.Implementation.MiPOD_ORIGINAL
MiPOD_FIX_WET = mipod.Implementation.MiPOD_FIX_WET
DISTORTION_LIMITED_SENDER = simulate.DISTORTION_LIMITED_SENDER
PAYLOAD_LIMITED_SENDER = simulate.PAYLOAD_LIMITED_SENDER
PLS = PAYLOAD_LIMITED_SENDER
DLS = DISTORTION_LIMITED_SENDER
DiLS = DISTORTION_LIMITED_SENDER
ATTACKER_INDIFFERENT = tools.ATTACKER_INDIFFERENT
ATTACKER_OMNISCIENT = tools.ATTACKER_OMNISCIENT
BACKEND_PYTHON = tools.BACKEND_PYTHON
BACKEND_RUST = tools.BACKEND_RUST
# BINARY_SEARCH = simulate.BINARY_SEARCH
# NEWTON = simulate.NEWTON
# POLYNOMIAL_PROXY = simulate.POLYNOMIAL_PROXY
# TAYLOR_INVERSE = simulate.TAYLOR_INVERSE
# SOLVER_BSEARCH_DDE = simulate.SOLVER_BSEARCH_DDE
# SOLVER_BSEARCH = simulate.SOLVER_BSEARCH
# SOLVER_NEWTON = simulate.SOLVER_NEWTON
# SOLVER_POLYPROXY = simulate.SOLVER_POLYPROXY
# TAYLOR_NEWTON = simulate.TAYLOR_NEWTON

# set version
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # for Python < 3.8
try:
    __version__ = version("conseal")
except PackageNotFoundError:
    __version__ = None

__all__ = [
    'F5',
    'hill',
    'hugo',
    'jmipod',
    'juniward',
    'lsb',
    'mipod',
    'nsF5',
    'uerd',
    'wow',
    'sunigard',
    'suniward',
    'simulate',
    'simulate_old',
    'tools',
    'JUNIWARD_ORIGINAL',
    'JUNIWARD_FIX_OFF_BY_ONE',
    'EBS_ORIGINAL',
    'EBS_FIX_WET',
    'LSB_REPLACEMENT',
    'LSB_MATCHING',
    '__version__',
]
