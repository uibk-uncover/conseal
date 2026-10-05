
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sys
import tempfile
import time
import unittest
import warnings

from . import defs
from . import stegolab_hill


class TestHILL(unittest.TestCase):
    """Test suite for HILL embedding."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_cost(self, fname: str):
        self._logger.info(f'TestHILL.test_cost({fname})')
        # load cover
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        # simulate the stego
        alpha = .4
        rho_p1, rho_m1 = cl.hill.compute_cost_adjusted(x)
        (p_p1, p_m1), lbda = cl.simulate.probability(
            rhos=(rho_p1, rho_m1),
            alpha=alpha,
            n=x.size,
        )
        # estimate average relative payload
        Hp = cl.simulate.entropy((rho_p1, rho_m1), ps=(p_p1, p_m1))
        alpha_hat = Hp/x.size
        self.assertAlmostEqual(alpha, alpha_hat, 3)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_simulate(self, fname: str):
        self._logger.info(f'TestHILL.test_simulate({fname})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))

        # simulate the stego
        alpha = .4
        x1 = cl.hill.simulate_single_channel(x0, alpha, seed=12345)

        # check changes
        self.assertEqual(x1.dtype, np.uint8)
        np.testing.assert_array_equal(x0.shape, x1.shape)
        delta = x0.astype('int16') - x1.astype('int16')
        self.assertLessEqual(delta.max(), 1)
        self.assertGreaterEqual(delta.min(), -1)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_stegolab_probability(self, fname: str):
        self._logger.info(f'TestHILL.test_stegolab_probability({fname=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        alpha = .4
        # stegolab, as in HILL.hide()
        wet_cost = 10**10
        rho = stegolab_hill.hill_cost(x0)
        rho[np.isnan(rho)] = wet_cost
        rho[rho > wet_cost] = wet_cost
        rho_p1, rho_m1 = rho.copy(), rho.copy()
        rho_p1[x0 == 255] = wet_cost
        rho_m1[x0 == 0] = wet_cost
        lbda_ref = stegolab_hill.calc_lambda(rho_p1, rho_m1, alpha * x0.size, x0.size)
        denum = 1 + np.exp(-lbda_ref * rho_p1) + np.exp(-lbda_ref * rho_m1)
        ps_ref = np.exp(-lbda_ref * rho_p1) / denum, np.exp(-lbda_ref * rho_m1) / denum
        # conseal
        ps, _ = cl.simulate.probability(cl.hill.compute_cost_adjusted(x0), alpha)
        # same change probabilities, also in flat areas
        np.testing.assert_allclose(ps, ps_ref, atol=1e-9)
        """
        We compare the change probabilities rather than the costs,
        because stegolab differs in several aspects

        - the residual is not divided by 4, so the costs are 4 times lower
        - the flat areas are regularized by adding 1e-10, instead of clipping

        This makes the costs differ in flat areas.
        To avoid heuristics, we instead compare in the probability space.
        """

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_rust(self, fname: str):
        self._logger.info(f'TestHILL.test_rust({fname=})')
        #
        with cl.BACKEND_PYTHON:
            path = defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'
            x0 = np.array(Image.open(path))
            rho_ref = cl.hill._costmap.compute_cost(x0)
        with cl.BACKEND_RUST:
            path = defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'
            x0 = np.array(Image.open(path))
            rho = cl.hill._costmap.compute_cost(x0)
        #
        np.testing.assert_allclose(rho, rho_ref, rtol=1e-12)


    @parameterized.expand([[f] for f in defs.TEST_IMAGES[:2]])
    def test_fallback(self, fname: str):
        self._logger.info(f'TestHILL.test_fallback({fname=})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        with cl.BACKEND_RUST:
            # uint8 cover, Rust without warning
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                rho = cl.hill._costmap.compute_cost(x0)
            # other dtype, falls back to Python with warning
            with self.assertWarns(RuntimeWarning):
                rho_float = cl.hill._costmap.compute_cost(x0.astype('float64'))
            np.testing.assert_allclose(rho_float, rho, rtol=1e-10)

    def test_speedup(self):
        self._logger.info('TestHILL.test_speedup()')
        # load covers once, time only the cost computation
        x0s = [
            np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
            for fname in defs.TEST_IMAGES
        ]
        for name, compute_cost in [
            ('rust', lambda x0: cl.hill._costmap.compute_cost(x0)),
            ('python separable', lambda x0: cl.hill._costmap._compute_cost(x0, separable=True)),
            ('python non-separable', lambda x0: cl.hill._costmap._compute_cost(x0, separable=False)),
        ]:
            with cl.BACKEND_RUST if name == 'rust' else cl.BACKEND_PYTHON:
                start = time.perf_counter()
                for x0 in x0s:
                    compute_cost(x0)
                end = time.perf_counter()
            print(f'HILL {name}: {end - start:.3f} s for {len(x0s)} images')
