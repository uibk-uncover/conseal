"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""
import conseal as cl
import jpeglib
import logging
import numpy as np
from parameterized import parameterized
import time
import unittest
import warnings
from . import defs


STEGO_DIR = defs.ASSETS_DIR / 'uerd'


class TestUERD(unittest.TestCase):
    """Test suite for UERD embedding."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([
        ("seal1.jpg", 0.4, 1, "seal1_alpha_0.4_seed_1.jpg"),
        ("seal2.jpg", 0.4, 2, "seal2_alpha_0.4_seed_2.jpg"),
        ("seal3.jpg", 0.4, 3, "seal3_alpha_0.4_seed_3.jpg"),
        ("seal4.jpg", 0.4, 4, "seal4_alpha_0.4_seed_4.jpg"),
        ("seal5.jpg", 0.4, 5, "seal5_alpha_0.4_seed_5.jpg"),
        ("seal6.jpg", 0.4, 6, "seal6_alpha_0.4_seed_6.jpg"),
        ("seal7.jpg", 0.4, 7, "seal7_alpha_0.4_seed_7.jpg"),
        ("seal8.jpg", 0.4, 8, "seal8_alpha_0.4_seed_8.jpg"),
    ])
    def test_matlab_equivalence(self, cover_filepath, embedding_rate, seed, stego_filepath):
        self._logger.info(
            f'TestUERD.test_matlab_equivalence('
            f'{cover_filepath=}, {stego_filepath=}, {embedding_rate=}, {seed=})'
        )

        # Read cover image
        im0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / cover_filepath)
        y0 = im0.Y

        # cl.uerd.simulate_single_channel differs from the Matlab implementation, because it uses another random number generator and the random numbers are arranged differently.
        # To show equivalence to the Matlab implementation, we need to use the Marsenne Twister and arrange the random numbers in the same order, as shown below.

        # Simulate embedding
        rho_p1, rho_m1 = cl.uerd.compute_cost_adjusted(y0, im0.qt[0], wet_cost=10 ** 13)

        # Rearrange from [num_vertical_blocks, num_horizontal_blocks, 8, 8] to [num_vertical_blocks * 8, num_horizontal_blocks * 8]
        rho_p1_2d = cl.tools.dct.jpeglib_to_jpegio(rho_p1)
        rho_m1_2d = cl.tools.dct.jpeglib_to_jpegio(rho_m1)
        y0_2d = cl.tools.dct.jpeglib_to_jpegio(y0)

        # Transform costmaps into embedding probability maps
        n = cl.tools.dct.nzAC(y0)
        ps, lbda = cl.simulate.probability(
            rhos=(rho_p1_2d, rho_m1_2d),
            alpha=embedding_rate,
            n=n,
        )

        # Simulate embedding
        delta_2d = cl.simulate.sample(
            ps,
            generator='MT19937',
            seed=seed,
        )

        # Add embedding changes onto the cover DCT coefficients
        y1_2d = y0_2d + delta_2d
        y1 = cl.tools.dct.jpegio_to_jpeglib(y1_2d)

        # Read stego image created using the Matlab implementation
        matlab_stego_im = jpeglib.read_dct(STEGO_DIR / stego_filepath)

        # Compare Matlab stego to our stego DCT coefficients
        np.testing.assert_array_equal(y1, matlab_stego_im.Y)

    @parameterized.expand([[.05], [.1], [.2], [.4]])
    def test_simulate_uerd_grayscale(self, alpha: float):
        self._logger.info(f'TestUERD.test_simulate_uerd_grayscale({alpha=})')

        # Load cover
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / 'seal6.jpg')

        # Initialize stego
        jpeg1 = jpeg0.copy()

        # Simulate the stego
        rhos = cl.uerd.compute_cost_adjusted(jpeg0.Y, jpeg0.qt[0])
        ps, lbda = cl.simulate.probability(
            rhos=rhos,
            alpha=alpha,
            n=jpeg0.Y.size,
        )
        jpeg1.Y += cl.simulate.sample(
            ps,
            seed=12345,
        )

        # Estimate average relative payload
        Hx = cl.simulate.entropy(
            rhos,
            ps=ps,
            q=3,
        )

        alpha_hat = Hx / jpeg0.Y.size
        self.assertAlmostEqual(alpha, alpha_hat, 3)

        # Compute embedding rate bound
        alpha_bound = cl.tools.dct.embedding_rate(jpeg0.Y, jpeg1.Y, q=3)
        self.assertLess(alpha, alpha_bound)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_rust(self, fname: str):
        self._logger.info(f'TestUERD.test_rust({fname=})')
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        with cl.BACKEND_PYTHON:
            rho_ref = cl.uerd._costmap.compute_cost(jpeg0.Y, jpeg0.qt[0])
        with cl.BACKEND_RUST:
            rho = cl.uerd._costmap.compute_cost(jpeg0.Y, jpeg0.qt[0])
        # integer arithmetic up to the division, bit-identical
        np.testing.assert_array_equal(rho, rho_ref)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES[:2]])
    def test_fallback(self, fname: str):
        self._logger.info(f'TestUERD.test_fallback({fname=})')
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        with cl.BACKEND_RUST:
            # int16 DCT, Rust without warning
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                rho = cl.uerd._costmap.compute_cost(jpeg0.Y, jpeg0.qt[0])
            # other dtype, falls back to Python with warning
            with self.assertWarns(RuntimeWarning):
                rho_float = cl.uerd._costmap.compute_cost(jpeg0.Y.astype('float64'), jpeg0.qt[0])
            with self.assertWarns(RuntimeWarning):
                rho_qt = cl.uerd._costmap.compute_cost(jpeg0.Y, jpeg0.qt[0].astype('int32'))
        np.testing.assert_array_equal(rho_float, rho)
        np.testing.assert_array_equal(rho_qt, rho)

    def test_rust_zero_energy(self):
        self._logger.info('TestUERD.test_rust_zero_energy()')
        # blocks with zero AC energy in the neighborhood get infinite cost
        jpeg0 = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{defs.TEST_IMAGES[0]}.jpg')
        y0 = np.zeros_like(jpeg0.Y)
        y0[0, 0, 0, 1] = 5  # single non-zero AC, energy only around block (0, 0)
        with cl.BACKEND_PYTHON:
            rho_ref = cl.uerd._costmap.compute_cost(y0, jpeg0.qt[0])
        with cl.BACKEND_RUST:
            rho = cl.uerd._costmap.compute_cost(y0, jpeg0.qt[0])
        self.assertTrue(np.isinf(rho).any())
        np.testing.assert_array_equal(rho, rho_ref)

    def test_speedup(self):
        self._logger.info('TestUERD.test_speedup()')
        # load covers once, time only the cost computation
        jpegs = [
            jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
            for fname in defs.TEST_IMAGES
        ]
        for name, backend in [('rust', cl.BACKEND_RUST), ('python', cl.BACKEND_PYTHON)]:
            with backend:
                start = time.perf_counter()
                for jpeg0 in jpegs:
                    cl.uerd._costmap.compute_cost(jpeg0.Y, jpeg0.qt[0])
                end = time.perf_counter()
            print(f'UERD {name}: {end - start:.3f} s for {len(jpegs)} images')


__all__ = ["TestUERD"]
