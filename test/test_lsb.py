
import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from PIL import Image
from parameterized import parameterized
import sys
import tempfile
import time
import unittest

from . import defs
STEGO_DIR = defs.ASSETS_DIR / 'lsb'


class TestLSB(unittest.TestCase):
    """Test suite for LSB embedding."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([
        [alpha, modify, locate]
        for alpha in [.05, .1, .2, .4]
        for modify in [cl.LSB_REPLACEMENT, cl.LSB_MATCHING]
        for locate in [cl.LOCATION_PERMUTED, cl.LOCATION_SEQUENTIAL]
    ])
    def test_simulate(self, alpha: float, modify: str, locate: cl.Location):
        self._logger.info(f"TestLSB.test_simulate({alpha}, {locate})")
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / 'seal1.png'))
        # embed steganography
        start = time.perf_counter()
        x1 = cl.lsb.simulate(
            x0, alpha,
            modify=modify,
            locate=locate,
            seed=12345,
        )
        end = time.perf_counter()
        # test range
        self.assertLessEqual((x1 - x0.astype('int32')).max(), 1)
        self.assertLessEqual((x1 - x0.astype('int32')).min(), -1)
        # test changes
        if modify == cl.LSB_REPLACEMENT:
            self.assertTrue((x0[x0 < x1] % 2 == 0).all())
            self.assertTrue((x0[x0 > x1] % 2 == 1).all())
        # test change rate
        beta_hat = (x0 != x1).mean()
        alpha_hat = beta_hat * 2
        self.assertAlmostEqual(alpha, alpha_hat, 2)
        # test speed
        delta = end - start
        self.assertLess(delta, .10)  # faster than 100ms
        self._logger.info(f'LSB {modify} embedding [{locate=}, {modify=}] 0.4bpnzAC in 512x512: {delta*1000:.02f} ms')

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_cost(self, f):
        self._logger.info(f'TestLSB.test_cost({f=})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # embed steganography
        rhos = cl.lsb.compute_cost_adjusted(x0)
        seed = cl.tools.password_to_seed(f)
        delta = cl.simulate.ternary(rhos=rhos, alpha=.4, n=x0.size, e=2, seed=seed)
        # test change rate
        self.assertAlmostEqual(.4/2, (delta != 0).mean(), 2)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_probability(self, f):
        self._logger.info(f'TestLSB.test_probability({f=})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # embed steganography
        ps, _ = cl.lsb._costmap.probability(x0, alpha=.4)
        delta = cl.simulate._ternary.simulate(ps=ps, seed=12345)
        # test change rate
        self.assertAlmostEqual((delta != 0).mean(), .4/2, 2)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_dct(self, f):
        self._logger.info(f'TestLSB.test_dct({f=})')
        # load cover
        y0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{f}.jpg').Y
        # simulate the stego
        y1 = cl.lsb.simulate(
            y0, .4,
            modify=cl.LSB_MATCHING,
            locate=cl.LOCATION_PERMUTED,
            cover_range=(-1024, 1023),
            seed=12345,
        )
        # check change rate
        beta_hat = (y0 != y1).mean()
        alpha_hat = beta_hat * 2
        self.assertAlmostEqual(.4, alpha_hat, 2)

    # TODO: chi2 test
    # TODO: ws

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_selected(self, f):
        self._logger.info(f'TestLSB.test_selected({f=})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
        # get cost
        rhos = cl.ws._costmap.compute_cost(x0)
        # simulate the stego
        x1 = cl.lsb.simulate(x0, alpha=.1, locate=cl.LOCATION_SELECTED, rhos=rhos, seed=12345)
        x1_ = cl.lsb.simulate(x0, alpha=.1, locate=cl.LOCATION_PERMUTED, seed=12345)
        #
        E_rho = np.mean((x1 != x0).astype('float') * rhos)
        E_rho_ = np.mean((x1_ != x0).astype('float') * rhos)
        self.assertLess(E_rho, E_rho_ * .1)  # 10 fold smaller (it reduces the WS cost)


__all__ = ["TestLSB"]
