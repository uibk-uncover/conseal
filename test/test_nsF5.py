"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from parameterized import parameterized
import tempfile
import unittest
from . import defs


STEGO_DIR = defs.ASSETS_DIR / 'nsF5'


class TestnsF5(unittest.TestCase):
    """Test suite for nsF5 embedding."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f, .4, i+1] for i, f in enumerate(defs.TEST_IMAGES)])
    def test_simulate(self, fname, alpha, seed):
        self._logger.info('TestnsF5.test_simulate()')
        #
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        y1 = cl.nsF5.simulate_single_channel(
            y0=jpeg0.Y,
            alpha=alpha,
            seed=seed,
        )
        #
        jpeg1_ref = jpeglib.read_dct(STEGO_DIR / f'{fname}_alpha_{alpha}_seed_{seed}.jpg')
        np.testing.assert_array_equal(y1, jpeg1_ref.Y)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_probability(self, fname):
        self._logger.info('TestnsF5.test_probability()')
        # load cover
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        # embed steganography
        ps, _ = cl.nsF5._costmap.probability(y0=jpeg0.Y, alpha=.4)
        delta = cl.simulate._ternary.simulate(ps=ps, seed=12345)
        # direct simulation as reference
        delta_ref = jpeg0.Y - cl.nsF5.simulate_single_channel(
            y0=jpeg0.Y,
            alpha=.4,
            seed=12345,
        )
        self.assertAlmostEqual((delta != 0).mean(), (delta_ref != 0).mean(), 3)
        # self.assertLess((delta != 0).mean() - (delta_ref != 0).mean(), 1e-3)
        # TODO: check where the changes are

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_cost(self, fname):
        self._logger.info('TestnsF5.test_cost()')
        # load cover
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        # embed steganography
        rhos = cl.nsF5.compute_cost_adjusted(jpeg0.Y)
        delta = cl.simulate.ternary(
            rhos=rhos,
            alpha=.4,
            n=cl.tools.nzAC(jpeg0.Y),
            seed=12345,
        ).astype('int16')
        # direct simulation as reference
        delta_ref = jpeg0.Y - cl.nsF5.simulate_single_channel(
            y0=jpeg0.Y,
            alpha=.4,
            seed=12345,
        )
        self.assertAlmostEqual((delta != 0).mean(), (delta_ref != 0).mean(), 3)
        # self.assertLess((delta != 0).mean() - (delta_ref != 0).mean(), 1e-3)
        # TODO: check where the changes are

    @parameterized.expand([
        [fname, alpha]
        for fname in defs.TEST_IMAGES
        for alpha in [0, .1, .4]
    ])
    def test_F5(self, fname: str, alpha: float):
        self._logger.info(f"TestnsF5.test_F5_histogram({fname}, {alpha})")
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        y0 = jpeg0.Y
        y1 = cl.F5.simulate_single_channel(y0, alpha, seed=12345)
        beta = (y0 != y1).sum() / cl.tools.nzAC(y0)

        # cartesian calibration
        jpeglib.from_dct(Y=y1, qt=jpeg0.qt).write_dct(self.tmp.name)
        x1 = jpeglib.read_spatial(self.tmp.name).spatial
        x2 = x1[3:-5, 3:-5]  # crop 4x4
        jpeglib.from_spatial(x2).write_spatial(self.tmp.name, qt=jpeg0.qt)
        y2 = jpeglib.read_dct(self.tmp.name).Y

        #
        h1_01, _ = np.histogram(y1[:, :, 0, 1].flatten(), 8, range=(-4, 4))
        h1_10, _ = np.histogram(y1[:, :, 1, 0].flatten(), 8, range=(-4, 4))
        h1_11, _ = np.histogram(y1[:, :, 1, 1].flatten(), 8, range=(-4, 4))
        h2_01, _ = np.histogram(y2[:, :, 0, 1].flatten(), 8, range=(-4, 4))
        h2_10, _ = np.histogram(y2[:, :, 1, 0].flatten(), 8, range=(-4, 4))
        h2_11, _ = np.histogram(y2[:, :, 1, 1].flatten(), 8, range=(-4, 4))
        #
        beta_01 = (h1_01[0+4] - h2_01[0+4]) / (h2_01[-1+4] + h2_01[1+4])
        beta_10 = (h1_10[0+4] - h2_10[0+4]) / (h2_10[-1+4] + h2_10[1+4])
        beta_hat = np.mean([beta_01, beta_10])
        beta_hat = np.clip(beta_hat, 0, None)
        # print(beta_hat, beta, '|', np.abs(beta - beta_hat))
        # print('\n', np.abs(beta - beta_hat))

        # print(beta, beta_hat)
        self.assertLess(np.abs(beta - beta_hat), .08)


__all__ = ["TestnsF5"]
