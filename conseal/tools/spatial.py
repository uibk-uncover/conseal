"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import numpy as np
import typing

from .common import entropy


# def embeddable_pixels(
#     x: np.ndarray = None,
#     wet: typing.Tuple = tuple()
# ) -> int:
#     """Get number of embeddable pixels."""
#     # wet elements
#     excluded = sum([(x == w).sum() for w in tuple(wet)])
#     # embeddable elements
#     embeddable = np.prod(x.shape) - excluded
#     return embeddable


# def change_rate(
#     x_c: np.ndarray,
#     x_s: np.ndarray,
#     q: int = 3
# ) -> float:
#     """Estimate change rate in spatial domain"""
#     # no changes
#     if (x_c == x_s).all():
#         return 0.
#     # embeddable elements
#     embeddable = embeddable_pixels(x_c)
#     # binary bound
#     if q == 2:
#         # number of changes
#         changes = (x_c != x_s).sum()
#         # change rate
#         betas = [changes/embeddable]
#     # ternary bound
#     elif q == 3:
#         # number o
#         # number of changes
#         changesP1 = (x_c < x_s).sum()
#         changesM1 = (x_c > x_s).sum()
#         # change rate
#         betas = [
#             changesP1/embeddable,
#             changesM1/embeddable,
#         ]

#     else:
#         raise NotImplementedError(f'{q}-ary code not implemented')

#     return betas


# def embedding_rate(
#     x_c: np.ndarray,
#     x_s: np.ndarray,
#     q: int = 3
# ) -> float:
#     """Estimate embedding rate in spatial domain"""
#     # no changes
#     if (x_c == x_s).all():
#         return 0.
#     # change rates
#     betas = change_rate(x_c, x_s, q=q)
#     # embedding rate
#     alpha = entropy(*betas)
#     return alpha


def daubechies8(
) -> typing.Tuple[typing.Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Returns Daubechies filter bank."""
    # 1D wavelet filters (Daubechies 8)
    hpdf = np.array([
        -0.0544158422, +0.3128715909, -0.6756307363, +0.5853546837,
        +0.0158291053, -0.2840155430, -0.0004724846, +0.1287474266,
        +0.0173693010, -0.0440882539, -0.0139810279, +0.0087460940,
        +0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ])
    lpdf = np.power(-1, np.arange(len(hpdf)))*np.flip(hpdf)
    # 2D wavelet filters
    F = [
        lpdf.reshape(-1, 1) @ hpdf.reshape(1, -1),
        hpdf.reshape(-1, 1) @ lpdf.reshape(1, -1),
        hpdf.reshape(-1, 1) @ hpdf.reshape(1, -1),
    ]
    return (hpdf, lpdf), F
