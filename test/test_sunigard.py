
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import scipy.io
import unittest
# from scipy.io import loadmat

from . import defs

# STEGO_DIR = defs.ASSETS_DIR / 'suniward'
# COST_DIR = STEGO_DIR / 'costmap-matlab'


class TestSUNIGARD(unittest.TestCase):
    """Test suite for S-UNIGARD embedding."""
    _logger = logging.getLogger(__name__)

    def test_gabor(self):
        self._logger.info(f'TestSUNIGARD.test_gabor()')
        features = cl.sunigard._costmap.gabor()
        self.assertEqual(len(features), 32)
        for f in features:
            self.assertEqual(f.shape, (11, 11))

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_suniward_matlab(self, f):
        self._logger.info(f'TestSUNIGARD.test_compare_suniward_matlab({f})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # calculate cost
        rho = cl.sunigard._costmap.compute_cost(x0=x0)

        # no reference for S-UNIGARD :(

    @parameterized.expand([
        [fname, i+1]
        for i, fname in enumerate(defs.TEST_IMAGES)
    ])
    def test_compare_embedding_matlab(self, fname: str, seed: int):
        self._logger.info(f'TestSUNIGARD.test_compare_wow_matlab({fname}, {seed})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        # embed steganography
        x1 = cl.sunigard.simulate_single_channel(
            x0=x0,
            alpha=.4,
            generator="MT19937",
            order='F',
            seed=seed,
        )

        # no reference for S-UNIGARD :(



__all__ = ["TestSUNIGARD"]
