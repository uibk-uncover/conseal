
import pkg_resources

#
from . import coding
from . import simulate
from . import tools

# spatial
from . import hill
from . import hugo
from . import lsb  # can be also used for JPEG
from . import mipod
from . import suniward
from . import wow

# JPEG
from . import ebs
from . import F5
from . import juniward
from . import nsF5
from . import uerd


# abbreviations of enum
JUNIWARD_ORIGINAL = juniward.Implementation.JUNIWARD_ORIGINAL
JUNIWARD_FIX_OFF_BY_ONE = juniward.Implementation.JUNIWARD_FIX_OFF_BY_ONE
EBS_ORIGINAL = ebs.Implementation.EBS_ORIGINAL
EBS_FIX_WET = ebs.Implementation.EBS_FIX_WET
LSB_REPLACEMENT = lsb.Change.LSB_REPLACEMENT
LSB_MATCHING = lsb.Change.LSB_MATCHING
MiPOD_ORIGINAL = mipod.Implementation.MiPOD_ORIGINAL
MiPOD_FIX_WET = mipod.Implementation.MiPOD_FIX_WET
DISTORTION_LIMITED_SENDER = simulate.DISTORTION_LIMITED_SENDER
PAYLOAD_LIMITED_SENDER = simulate.PAYLOAD_LIMITED_SENDER
PLS = PAYLOAD_LIMITED_SENDER
DLS = DISTORTION_LIMITED_SENDER
ATTACKER_INDIFFERENT = tools.ATTACKER_INDIFFERENT
ATTACKER_OMNISCIENT = tools.ATTACKER_OMNISCIENT

# package version
try:
    __version__ = pkg_resources.get_distribution("conseal").version
except pkg_resources.DistributionNotFound:
    __version__ = None

__all__ = [
    'F5',
    'hill',
    'hugo',
    'juniward',
    'lsb',
    'mipod',
    'nsF5',
    'uerd',
    'wow',
    'simulate',
    'tools',
    'JUNIWARD_ORIGINAL',
    'JUNIWARD_FIX_OFF_BY_ONE',
    'EBS_ORIGINAL',
    'EBS_FIX_WET',
    'LSB_REPLACEMENT',
    'LSB_MATCHING',
    '__version__',
]
