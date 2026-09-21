"""
J-MiPOD implementation based on https://codeocean.com/capsule/7800700/tree/v4
Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck

This Python implementation is ported from the original Matlab implementation provided by the paper authors. Please find the license of the original implementation below.
-------------------------------------------------------------------------
Copyright (c) 2020 Remi Cogranne, UTT (Troyes University of Technology). All Rights Reserved.
-------------------------------------------------------------------------
This code is provided by the author under Creative Common License (CC BY-NC-SA 4.0) which, as explained on this webpage https://creativecommons.org/licenses/by-nc-sa/4.0/ Allows modification, redistribution, provided that:
* You share your code under the same license;
* That you give credits to the authors;
* The code is used only for non-commercial purposes (which includes education and research)
-------------------------------------------------------------------------
The authors hereby grant the use of the present code without fee, and without a written agreement under compliance with aforementioned and provided and the present copyright notice appears in all copies. The program is supplied "as is," without any accompanying services from the UTT or the authors. The UTT does not warrant the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason. In no event shall the UTT or the authors be liable to any party for any consequential damages. The authors also fordid any practical use of this code for communication by hiding data into JPEG images.
-------------------------------------------------------------------------

J-MiPOD shares its Lagrangian-multiplier search and lookup-table inversion with (spatial) MiPOD,
since both methods derive the change rate from the same Fisher information -> ternary-entropy relationship.
These domain-agnostic pieces (including the precomputed ixlnx3.mat lookup table) are therefore
reused directly from :mod:`conseal.mipod._defs` instead of being duplicated here.
"""

from ..mipod._defs import (
    wiener2,
    im2col,
    pad_copy_borders,
    invxlnx3_fast,
    ternary_entropy,
    ternary_probs,
    prepare_lookup_table,
    load_lookup_table,
)

__all__ = [
    'wiener2',
    'im2col',
    'pad_copy_borders',
    'invxlnx3_fast',
    'ternary_entropy',
    'ternary_probs',
    'prepare_lookup_table',
    'load_lookup_table',
]
