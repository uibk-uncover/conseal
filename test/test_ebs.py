
import conseal
import jpeglib
import logging
import numpy as np
from parameterized import parameterized
import scipy.io
import unittest

from .defs import ASSETS_DIR
STEGO_DIR = ASSETS_DIR / 'ebs'
COVER_DIR = ASSETS_DIR / 'cover'


class TestEBS(unittest.TestCase):
    """Test suite for EBS embedding."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([
        ('mountain',),
        ('lizard',),
        ('nuclear',),
    ])
    def test_compute_cost(self, fname):
        self._logger.info(f'TestEBS.test_compute_cost({fname})')
        # load cover
        jpeg = jpeglib.read_dct(COVER_DIR / f'{fname}_gray.jpeg')

        # compute cost
        rho, _ = conseal.ebs.compute_cost_adjusted(jpeg.Y, jpeg.qt[0])

        # convert to compare
        rho = conseal.tools.dct.jpeglib_to_jpegio(rho)

        # compare to Remi Cogranne's Matlab implementation
        remi = scipy.io.loadmat(f'test/assets/ebs/{fname}_gray_costmap.mat',
                                simplify_cells=True)
        rho_remi = remi[f'rho_{fname}']
        self.assertTrue(np.allclose(rho, rho_remi))

    @parameterized.expand([
        ('mountain',),
        ('lizard',),
        ('nuclear',),
    ])
    def test_simulate(self, fname):
        self._logger.info(f'TestEBS.test_simulate({fname})')
        # load cover
        jpeg = jpeglib.read_dct(COVER_DIR / f'{fname}_gray.jpeg')
        Yc = jpeg.Y

        # simulate
        Ys = conseal.ebs.simulate_single_channel(
            Yc,
            jpeg.qt[0],
            .4,
            seed=12345,
        )


__all__ = ["TestEBS"]
