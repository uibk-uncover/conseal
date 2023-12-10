
import pkg_resources

from . import ebs
from . import juniward
from . import nsF5
from . import uerd
from . import simulate
from . import tools

# abbreviations of enum
JUNIWARD_ORIGINAL = juniward.Implementation.JUNIWARD_ORIGINAL
JUNIWARD_FIX_OFF_BY_ONE = juniward.Implementation.JUNIWARD_FIX_OFF_BY_ONE
EBS_ORIGINAL = ebs.Implementation.EBS_ORIGINAL
EBS_FIX_WET = ebs.Implementation.EBS_FIX_WET

# package version
try:
    __version__ = pkg_resources.get_distribution("conseal").version
except pkg_resources.DistributionNotFound:
    __version__ = None

__all__ = [
    'juniward',
    'nsF5',
    'uerd',
    'simulate',
    'tools',
    'JUNIWARD_ORIGINAL',
    'JUNIWARD_FIX_OFF_BY_ONE',
    '__version__',
]
