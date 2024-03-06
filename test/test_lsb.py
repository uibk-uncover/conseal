
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
        [alpha, modify, permute]
        for alpha in [.05, .1, .2, .4]
        for modify in [cl.LSB_REPLACEMENT, cl.LSB_MATCHING]
        for permute in [True, False]
    ])
    def test_simulate_LSB(self, alpha: float, modify: str, permute: bool):
        self._logger.info(f"TestLSB.test_simulate_LSB({alpha}, {modify}, {permute})")
        # load cover
        cover_spatial = np.array(Image.open(defs.COVER_UG_DIR / 'seal1.png'))
        # simulate the stego
        stego_spatial = cl.lsb.simulate(
            cover_spatial, alpha,
            modify=modify,
            permute=permute,
            seed=12345,
        )
        # check change rate
        beta_hat = (cover_spatial != stego_spatial).mean()
        alpha_hat = beta_hat * 2
        self.assertAlmostEqual(alpha, alpha_hat, 2)

    @parameterized.expand([
        [modify, permute]
        for modify in [cl.LSB_REPLACEMENT, cl.LSB_MATCHING]
        for permute in [True, False]
    ])
    def test_simulate_LSB_time(self, modify, permute):
        self._logger.info('TestLSB.test_simulate_LSB_time')
        # load cover
        cover_spatial = np.array(Image.open(defs.COVER_UG_DIR / 'seal1.png'))
        # time the simulation
        start = time.perf_counter()
        cl.lsb.simulate(
            cover_spatial, .4,
            modify=modify,
            permute=permute,
            seed=12345,
        )
        end = time.perf_counter()
        # test speed
        delta = end - start
        self.assertLess(delta, .10)  # faster than 100ms
        self._logger.info(f'LSB {modify} embedding [{permute=}, {modify=}] 0.4bpnzAC in 512x512: {delta*1000:.02f} ms')

    @parameterized.expand([
        [alpha, modify, permute]
        for alpha in [.05, .1, .2, .4]
        for modify in [cl.LSB_REPLACEMENT, cl.LSB_MATCHING]
        for permute in [True, False]
    ])
    def test_simulate_LSB_color(self, alpha: float, modify: str, permute: bool):
        self._logger.info(f"TestLSB.test_simulate_LSB_color({alpha}, {modify}, {permute})")
        # load cover
        cover_spatial = np.array(Image.open(defs.COVER_UC_DIR / 'seal1.png'))
        # simulate the stego
        stego_spatial = cl.lsb.simulate(
            cover_spatial, alpha,
            modify=modify,
            permute=permute,
            seed=12345,
        )
        # check change rate
        beta_hat = (cover_spatial != stego_spatial).mean()
        alpha_hat = beta_hat * 2
        self.assertAlmostEqual(alpha, alpha_hat, 2)

    @parameterized.expand([
        [alpha, modify, permute]
        for alpha in [.05, .1, .2, .4]
        for modify in [cl.LSB_REPLACEMENT, cl.LSB_MATCHING]
        for permute in [True, False]
    ])
    def test_simulate_LSB_dct(self, alpha: float, modify: str, permute: bool):
        self._logger.info('TestLSB.test_simulate_LSB_dct')
        # load cover
        dct_c = jpeglib.read_dct(defs.COVER_CG_DIR / 'seal1.jpg').Y
        # simulate the stego
        dct_s = cl.lsb.simulate(
            dct_c, alpha,
            modify=modify,
            permute=permute,
            cover_range=(-1024, 1023),
            seed=12345,
        )
        # check change rate
        beta_hat = (dct_c != dct_s).mean()
        alpha_hat = beta_hat * 2
        self.assertAlmostEqual(alpha, alpha_hat, 2)


    # TODO: chi2 test
    # TODO: ws


__all__ = ["TestLSB"]
