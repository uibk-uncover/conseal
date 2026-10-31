"""Conversion between messages and GF(q) symbols.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import math
import numpy as np


def num_symbols(length: int, q: int) -> int:
    """Number of GF(q) symbols to represent message of given length in bytes."""
    k = math.ceil(8 * length / math.log2(q))
    while q**k < 256**length:
        k += 1
    while k > 0 and q**(k-1) >= 256**length:
        k -= 1
    return k


def bytes_to_symbols(message: bytes, q: int) -> np.ndarray:
    """Converts bytes into GF(q) symbols, least significant first."""
    num = int.from_bytes(message, 'big')
    symbols = np.zeros(num_symbols(len(message), q), dtype='int64')
    for i in range(len(symbols)):
        num, symbols[i] = divmod(num, q)
    return symbols


def symbols_to_bytes(symbols: np.ndarray, length: int, q: int) -> bytes:
    """Converts GF(q) symbols into bytes, inverse of :func:`bytes_to_symbols`."""
    num = 0
    for symbol in reversed(symbols):
        num = num * q + int(symbol)
    # wrong key gives garbage, not error
    return (num % 256**length).to_bytes(length, 'big')
