
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import scipy.io
import tempfile
import time
import unittest
import warnings

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
        delta = cl.simulate.simulate(
            rhos=(rho_p1, rho_m1),
            alpha=.4,
            n=x0.size,
            generator='MT19937',
            order='F',
            seed=139187)
        x1 = x0 + delta

        # test costs against DDE matlab reference
        mat = scipy.io.loadmat(STEGO_DIR / f'costmap-matlab/{f}_costmap.mat')
        np.testing.assert_allclose(rho_p1, mat['rhoP1'], rtol=1e-10)
        np.testing.assert_allclose(rho_m1, mat['rhoM1'], rtol=1e-10)
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

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_rust(self, fname: str):
        self._logger.info(f'TestWOW.test_rust({fname=})')
        #
        with cl.BACKEND_PYTHON:
            path = defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'
            x0 = np.array(Image.open(path))
            rho_ref = cl.wow._costmap.compute_cost(x0)
        with cl.BACKEND_RUST:
            path = defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'
            x0 = np.array(Image.open(path))
            rho = cl.wow._costmap.compute_cost(x0)
        #
        np.testing.assert_allclose(rho, rho_ref, rtol=1e-12)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES[:2]])
    def test_fallback(self, fname: str):
        self._logger.info(f'TestWOW.test_fallback({fname=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        with cl.BACKEND_RUST:
            # uint8 cover, Rust without warning
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                rho = cl.wow._costmap.compute_cost(x0)
            # other dtype, falls back to Python with warning
            with self.assertWarns(RuntimeWarning):
                rho_float = cl.wow._costmap.compute_cost(x0.astype('float64'))
            np.testing.assert_allclose(rho_float, rho, rtol=1e-10)
        # non-separable filters, only in Python
        with self.assertWarns(RuntimeWarning):
            rho_2d = cl.wow._costmap.compute_cost(x0, separable=False)
        np.testing.assert_allclose(rho_2d, rho, rtol=1e-10)

    def test_speedup(self):
        self._logger.info('TestWOW.test_speedup()')
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
                    cl.wow._costmap.compute_cost(x0, **kw)
                end = time.perf_counter()
            print(f'WOW {name}: {end - start:.3f} s for {len(x0s)} images')

__all__ = ['TestWOW']
