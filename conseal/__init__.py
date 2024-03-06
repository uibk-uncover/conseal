
import pkg_resources

# spatial
from . import hugo
from . import lsb  # can be also used for JPEG

# JPEG
from . import ebs
from . import juniward
from . import nsF5
from . import uerd


#
from . import simulate
from . import tools

# abbreviations of enum
JUNIWARD_ORIGINAL = juniward.Implementation.JUNIWARD_ORIGINAL
JUNIWARD_FIX_OFF_BY_ONE = juniward.Implementation.JUNIWARD_FIX_OFF_BY_ONE
EBS_ORIGINAL = ebs.Implementation.EBS_ORIGINAL
EBS_FIX_WET = ebs.Implementation.EBS_FIX_WET
LSB_REPLACEMENT = lsb.Change.LSB_REPLACEMENT
LSB_MATCHING = lsb.Change.LSB_MATCHING

# package version
try:
    __version__ = pkg_resources.get_distribution("conseal").version
except pkg_resources.DistributionNotFound:
    __version__ = None

__all__ = [
    'hugo',
    'juniward',
    'lsb',
    'nsF5',
    'uerd',
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
