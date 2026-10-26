
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import scipy.io
import time
import unittest
import warnings
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

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_rust(self, fname: str):
        self._logger.info(f'TestSUNIGARD.test_rust({fname=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        with cl.BACKEND_PYTHON:
            rho_ref = cl.sunigard._costmap.compute_cost(x0)
        with cl.BACKEND_RUST:
            rho = cl.sunigard._costmap.compute_cost(x0)
        np.testing.assert_allclose(rho, rho_ref, rtol=1e-12)

    def test_fallback(self):
        self._logger.info('TestSUNIGARD.test_fallback()')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{defs.TEST_IMAGES[0]}.png'))
        with cl.BACKEND_RUST:
            # uint8 cover, Rust without warning
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                rho = cl.sunigard._costmap.compute_cost(x0)
            # other dtype, falls back to Python with warning
            with self.assertWarns(RuntimeWarning):
                rho_float = cl.sunigard._costmap.compute_cost(x0.astype('float64'))
        np.testing.assert_allclose(rho_float, rho, rtol=1e-12)

    def test_all_filters(self):
        self._logger.info('TestSUNIGARD.test_all_filters()')
        # cost sums over all 32 Gabor filters, not only the first three
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{defs.TEST_IMAGES[0]}.png'))
        filters = cl.sunigard._costmap.gabor()
        rho = cl.sunigard._costmap.compute_cost(x0)
        rho_sum = sum(cl.suniward._costmap.compute_cost(x0, filters=[f]) for f in filters)
        np.testing.assert_allclose(rho, rho_sum, rtol=1e-12)

    def test_speedup(self):
        self._logger.info('TestSUNIGARD.test_speedup()')
        # load covers once, time only the cost computation
        x0s = [
            np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
            for fname in defs.TEST_IMAGES
        ]
        for name, backend in [('rust', cl.BACKEND_RUST), ('python', cl.BACKEND_PYTHON)]:
            with backend:
                start = time.perf_counter()
                for x0 in x0s:
                    cl.sunigard._costmap.compute_cost(x0)
                end = time.perf_counter()
            print(f'S-UNIGARD {name}: {end - start:.3f} s for {len(x0s)} images')


__all__ = ["TestSUNIGARD"]
