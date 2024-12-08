
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

STEGO_DIR = defs.ASSETS_DIR / 'suniward'
COST_DIR = STEGO_DIR / 'costmap-matlab'


class TestSUNIWARD(unittest.TestCase):
    """Test suite for S-UNIWARD embedding."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compare_suniward_matlab(self, f):
        self._logger.info(f'TestSUNIWARD.test_compare_suniward_matlab({f})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # calculate cost
        rho = cl.suniward._costmap.compute_cost(x0=x0)

        # load matlab reference
        rho_matlab = scipy.io.loadmat(COST_DIR / f'{f}_costmap.mat')['rho']
        np.testing.assert_allclose(rho, rho_matlab)
        # self.assertTrue(np.allclose(costmap, costmap_matlab))

    @parameterized.expand([
        [fname, i+1]
        for i, fname in enumerate(defs.TEST_IMAGES)
    ])
    def test_compare_embedding_matlab(self, fname: str, seed: int):
        self._logger.info(f'TestWOW.test_compare_wow_matlab({fname}, {seed})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        # embed steganography
        x1 = cl.suniward.simulate_single_channel(
            x0=x0,
            alpha=.4,
            generator="MT19937",
            order='F',
            seed=seed,
        )

        # test against DDE matlab reference
        x1_matlab = np.array(Image.open(STEGO_DIR / f'{fname}_alpha_0.4_seed_{seed}.png'))
        np.testing.assert_allclose(x1, x1_matlab)



__all__ = ["TestSUNIWARD"]
