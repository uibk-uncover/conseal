
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
        self._logger.info(f'TestSUNIWARD.test_compare_wow_matlab({fname}, {seed})')
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

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_rust(self, fname: str):
        self._logger.info(f'TestSUNIWARD.test_rust({fname=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        with cl.BACKEND_PYTHON:
            rho_ref = cl.suniward._costmap.compute_cost(x0)
        with cl.BACKEND_RUST:
            rho = cl.suniward._costmap.compute_cost(x0)
        np.testing.assert_allclose(rho, rho_ref, rtol=1e-12)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES[:2]])
    def test_rust_filters(self, fname: str):
        self._logger.info(f'TestSUNIWARD.test_rust_filters({fname=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        daubechies = cl.tools.spatial.daubechies8()[1]
        gabor = cl.sunigard._costmap.gabor()
        for kw in [
            {'separable': False},  # 2D Daubechies
            {'filters': gabor[:4]},  # odd filters
            {'filters': daubechies[:1] + gabor[:2]},  # mixed sizes, even and odd
        ]:
            with cl.BACKEND_PYTHON:
                rho_ref = cl.suniward._costmap.compute_cost(x0, **kw)
            with cl.BACKEND_RUST:
                # all filters in Rust, without warning
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RuntimeWarning)
                    rho = cl.suniward._costmap.compute_cost(x0, **kw)
            np.testing.assert_allclose(rho, rho_ref, rtol=1e-12)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES[:2]])
    def test_fallback(self, fname: str):
        self._logger.info(f'TestSUNIWARD.test_fallback({fname=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        with cl.BACKEND_RUST:
            # uint8 cover, Rust without warning
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                rho = cl.suniward._costmap.compute_cost(x0)
            # other dtype, falls back to Python with warning
            with self.assertWarns(RuntimeWarning):
                rho_float = cl.suniward._costmap.compute_cost(x0.astype('float64'))
        np.testing.assert_allclose(rho_float, rho, rtol=1e-12)

    def test_sum_filters(self):
        self._logger.info('TestSUNIWARD.test_sum_filters()')
        # cost is the sum over all filters, also beyond three
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{defs.TEST_IMAGES[0]}.png'))
        filters = cl.sunigard._costmap.gabor()[:5]
        for backend in [cl.BACKEND_PYTHON, cl.BACKEND_RUST]:
            with backend:
                rho = cl.suniward._costmap.compute_cost(x0, filters=filters)
                rho_sum = sum(cl.suniward._costmap.compute_cost(x0, filters=[f]) for f in filters)
            np.testing.assert_allclose(rho, rho_sum, rtol=1e-12)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_separable(self, fname: str):
        self._logger.info(f'TestSUNIWARD.test_separable({fname=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        with cl.BACKEND_PYTHON:
            rho_2d = cl.suniward._costmap.compute_cost(x0, separable=False)
            rho = cl.suniward._costmap.compute_cost(x0, separable=True)
        np.testing.assert_allclose(rho, rho_2d, rtol=1e-12)

    def test_speedup(self):
        self._logger.info('TestSUNIWARD.test_speedup()')
        # load covers once, time only the cost computation
        x0s = [
            np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
            for fname in defs.TEST_IMAGES
        ]
        for name, backend, kw in [
            ('rust', cl.BACKEND_RUST, {}),
            ('python separable', cl.BACKEND_PYTHON, {'separable': True}),
            ('python non-separable', cl.BACKEND_PYTHON, {'separable': False}),
        ]:
            with backend:
                start = time.perf_counter()
                for x0 in x0s:
                    cl.suniward._costmap.compute_cost(x0, **kw)
                end = time.perf_counter()
            print(f'S-UNIWARD {name}: {end - start:.3f} s for {len(x0s)} images')


__all__ = ["TestSUNIWARD"]
