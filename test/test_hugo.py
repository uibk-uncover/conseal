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


COST_DIR = defs.ASSETS_DIR / 'hugo' / 'costmap_matlab'
STEGO_DIR = defs.ASSETS_DIR / 'hugo' / 'stego_matlab'


class TestHUGO(unittest.TestCase):
    """Test suite for HUGO embedding."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_hugo_costmap(self, fname: str):
        self._logger.info(f'TestHUGO.test_compare_hugo_matlab({fname})')
        # load cover
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        # simulate the stego
        # import time
        # start = time.perf_counter()
        rho_p1, rho_m1 = cl.hugo.compute_cost_adjusted(x)
        # end = time.perf_counter()
        # print(fname, end-start)

        # load matlab reference
        mat = scipy.io.loadmat(COST_DIR / f'{fname}.mat')

        np.testing.assert_allclose(rho_p1, mat['rhoP1'])
        np.testing.assert_allclose(rho_m1, mat['rhoM1'])

    # @parameterized.expand([['00001'], ['00002'], ['00003'], ['00004'], ['00005']])
    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_hugo_stego(self, fname: str):
        self._logger.info(f'TestHUGO.test_hugo_stego({fname})')
        # load cover
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))

        # simulate the stego
        rho_p1, rho_m1 = cl.hugo.compute_cost_adjusted(x)
        y = x + cl.simulate.ternary(
            rho_p1=rho_p1,
            rho_m1=rho_m1,
            # rhos=(rhoP1, rhoM1),
            alpha=.4,
            n=x.size,
            order='F',
            generator='MT19937',
            seed=139187,
        )

        # load matlab reference
        y_ref = np.array(Image.open(STEGO_DIR / f'{fname}.png'))
        np.testing.assert_allclose(y, y_ref)
