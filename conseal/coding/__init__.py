"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from . import hamming
from . import wpc
from . import trellis

from .. import tools


def efficiency(alpha: float) -> float:
    """Calculates the embedding efficiency at the bound.

    :param alpha: embedding rate, at bits per element
    :type alpha: float
    :return: embedding efficiency, at bits per element
    :rtype: float

    :Example:

    >>> e = cl.coding.efficiency(0.4)  # e at alpha=0.4
    """
    return alpha / tools.inv_entropy(alpha)
