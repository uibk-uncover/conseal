"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""
import conseal as cl
import jpeglib
import logging
import numpy as np
from parameterized import parameterized
import unittest
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
        ps, lbda = cl.simulate._ternary.probability(
            rhos=(rho_p1_2d, rho_m1_2d),
            alpha=embedding_rate,
            n=n,
        )

        # Simulate embedding
        delta_2d = cl.simulate._ternary.simulate(
            ps=ps,
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
        ps, lbda = cl.simulate._ternary.probability(
            rhos=rhos,
            alpha=alpha,
            n=jpeg0.Y.size,
        )
        jpeg1.Y += cl.simulate._ternary.simulate(
            ps=ps,
            seed=12345,
        )

        # Estimate average relative payload
        _, Hx = cl.simulate.average_payload(
            lbda=lbda,
            ps=ps,
            q=3
        )

        alpha_hat = Hx / jpeg0.Y.size
        self.assertAlmostEqual(alpha, alpha_hat, 3)

        # Compute embedding rate bound
        alpha_bound = cl.tools.dct.embedding_rate(jpeg0.Y, jpeg1.Y, q=3)
        self.assertLess(alpha, alpha_bound)


__all__ = ["TestUERD"]
