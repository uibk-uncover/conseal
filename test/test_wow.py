
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import scipy.io
import tempfile
import unittest

from . import defs

STEGO_DIR = defs.ASSETS_DIR / 'wow'


class TestWOW(unittest.TestCase):
    """Test suite for WOW embedding."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_cost(self, f):
        self._logger.info(f'TestWOW.test_cost({f})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # embed steganography
        rho_p1, rho_m1 = cl.wow.compute_cost_adjusted(x0)
        delta = cl.simulate.ternary(
            rhos=(rho_p1, rho_m1),
            alpha=.4,
            n=x0.size,
            generator='MT19937',
            order='F',
            seed=139187)
        x1 = x0 + delta

        # test costs against DDE matlab reference
        mat = scipy.io.loadmat(STEGO_DIR / f'costmap-matlab/{f}_costmap.mat')
        np.testing.assert_allclose(rho_p1, mat['rhoP1'], rtol=1e-5)
        np.testing.assert_allclose(rho_m1, mat['rhoM1'], rtol=1e-5)
        # test stego against DDE matlab reference
        x1_ref = np.array(Image.open(STEGO_DIR / f'{f}.png'))
        np.testing.assert_array_equal(x1, x1_ref)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_simulate(self, f):
        self._logger.info(f'TestWOW.test_simulate({f})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # embed steganography
        x1 = cl.wow.simulate_single_channel(
            x0=x0,
            alpha=.4,
            generator='MT19937',
            order='F',
            seed=139187)
        # test stego against DDE matlab reference
        x1_ref = np.array(Image.open(STEGO_DIR / f'{f}.png'))
        np.testing.assert_array_equal(x1, x1_ref)


__all__ = ['TestWOW']
