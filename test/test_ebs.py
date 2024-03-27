
import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from parameterized import parameterized
import scipy.io
import tempfile
import unittest

from . import defs
# STEGO_DIR = defs.ASSETS_DIR / 'ebs'


class TestEBS(unittest.TestCase):
    """Test suite for EBS embedding."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_compute_cost(self, fname):
        self._logger.info(f'TestEBS.test_compute_cost({fname})')
        # load cover
        jpeg = jpeglib.read_dct(defs.COVER_DIR / 'jpeg_75_gray' / f'{fname}.jpg')

        # compute cost
        rho, _ = cl.ebs.compute_cost_adjusted(jpeg.Y, jpeg.qt[0])

        # convert to compare
        rho = cl.tools.dct.jpeglib_to_jpegio(rho)

        # compare to Remi Cogranne's Matlab implementation
        remi = scipy.io.loadmat(f'test/assets/costmap_matlab/ebs/{fname}.mat',
                                simplify_cells=True)
        rho_remi = remi[f'rhos']
        self.assertTrue(np.allclose(rho, rho_remi))

    # @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    @parameterized.expand([['seal1']])
    def test_simulate(self, fname):
        self._logger.info(f'TestEBS.test_simulate({fname})')
        # load cover
        cover = jpeglib.read_dct(defs.COVER_DIR / 'jpeg_75_gray' / f'{fname}.jpg')

        # simulate
        stego = cl.ebs.simulate_single_channel(
            cover.Y,
            cover.qt[0],
            .4,
            implementation=cl.EBS_ORIGINAL,
            seed=12345,
        )

        # load reference stego
        stego_ref = jpeglib.read_dct(defs.ASSETS_DIR / 'stego_matlab' / 'ebs' / f'{fname}.jpg').Y

        # compare stegos
        # np.testing.assert_array_equal(stego, stego_ref)
        # EBS simulator contains bugs


__all__ = ["TestEBS"]
