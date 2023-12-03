"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import jpeglib
import logging
import numpy as np
from parameterized import parameterized
import tempfile
import unittest

from defs import ASSETS_DIR, COVER_DIR
STEGO_DIR = ASSETS_DIR / 'uerd'


class TestUERD(unittest.TestCase):
    """Test suite for UERD embedding."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([
        ("landscape_gray.jpeg", "landscape_gray_matlab_alpha_0.5_seed_12345.jpg", 0.5, 12345),
        ("lizard_gray.jpeg", "lizard_gray_matlab_alpha_0.4_seed_6789.jpg", 0.4, 6789),
        ("mountain_gray.jpeg", "mountain_gray_matlab_alpha_0.2_seed_6020.jpg", 0.2, 6020),
        ("nuclear_gray.jpeg", "nuclear_gray_matlab_alpha_0.1_seed_91058.jpg", 0.1, 91058),
    ])
    def test_matlab_equivalence(self, cover_filepath, stego_filepath, payload, seed):
        self._logger.info(
            f'TestUERD.test_matlab_equivalence('
            f'{cover_filepath=}, {stego_filepath=}, {payload=}, {seed=})'
        )

        # Read cover image
        cover_im = jpeglib.read_dct(COVER_DIR / cover_filepath)
        cover_dct_coeffs = cover_im.Y
        n = cl.tools.dct.nzAC(cover_dct_coeffs)

        # Simulate embedding
        rho_p1, rho_m1 = cl.uerd.compute_distortion(cover_im.Y, cover_im.qt[0], wet_cost=10**13)
        # Rearrange from [num_vertical_blocks, num_horizontal_blocks, 8, 8] to [num_vertical_blocks * 8, num_horizontal_blocks * 8]
        rho_p1_2d = cl.tools.dct.jpeglib_to_jpegio(rho_p1)
        rho_m1_2d = cl.tools.dct.jpeglib_to_jpegio(rho_m1)
        cover_dct_coeffs_2d = cl.tools.dct.jpeglib_to_jpegio(cover_dct_coeffs)

        #
        (pChangeP1, pChangeM1), lbda = cl.simulate._ternary.probability(
            rhoP1=rho_p1_2d,
            rhoM1=rho_m1_2d,
            alpha=payload,
            n=n,
        )
        delta_dct_coeffs_2d = cl.simulate._ternary.simulate(
            pChangeP1=pChangeP1,
            pChangeM1=pChangeM1,
            generator='MT19937',
            # order='F',
            seed=seed,
        )
        stego_dct_coeffs_2d = cover_dct_coeffs_2d + delta_dct_coeffs_2d
        stego_dct_coeffs = cl.tools.dct.jpegio_to_jpeglib(stego_dct_coeffs_2d)

        # Read stego image created using Matlab implementation
        matlab_stego_im = jpeglib.read_dct(STEGO_DIR / stego_filepath)

        # Compare Matlab stego to our stego DCT coefficients
        np.testing.assert_array_equal(stego_dct_coeffs, matlab_stego_im.Y)

    @parameterized.expand([[.05], [.1], [.2], [.4]])
    def test_simulate_uerd_grayscale(self, alpha: float):
        self._logger.info(f'TestUERD.test_simulate_uerd_grayscale(alpha={alpha})')
        # load cover
        jpeg_c = jpeglib.read_dct(COVER_DIR / 'lizard.jpeg')
        jpeg_s = jpeg_c.copy()
        # simulate the stego
        rhoP1, rhoM1 = cl.uerd.compute_distortion(jpeg_c.Y, jpeg_c.qt[0])
        (pChangeP1, pChangeM1), lbda = cl.simulate._ternary.probability(
            rhoP1, rhoM1,
            alpha=alpha,
            n=jpeg_c.Y.size,
        )
        jpeg_s.Y += cl.simulate._ternary.simulate(
            pChangeP1,
            pChangeM1,
            seed=12345,
        )
        # estimate average relative payload
        _, Hx = cl.simulate.average_payload(
            lbda=lbda,
            pP1=pChangeP1,
            pM1=pChangeM1,
            q=3
        )
        alpha_hat = Hx / jpeg_s.Y.size
        self.assertAlmostEqual(alpha, alpha_hat, 3)
        # compute embedding rate bound
        alpha_bound = cl.tools.dct.embedding_rate(jpeg_c.Y, jpeg_s.Y, q=3)
        self.assertLess(alpha, alpha_bound)

    @parameterized.expand([[.05], [.1], [.2], [.4]])
    def test_simulate_uerd_color_luminance(self, alpha: float):
        self._logger.info(f'TestUERD.test_simulate_uerd_grayscale(alpha={alpha})')
        # load cover
        cover = jpeglib.read_dct(COVER_DIR / 'lizard.jpeg')
        stego = cover.copy()

        stego_Y = cl.uerd.simulate_single_channel(
            cover.Y,
            cover.qt[cover.quant_tbl_no[0]],
            embedding_rate=alpha,
            payload_mode="bpnzAC",
            seed=12345)

        # Fails
        stego.Y = stego_Y
        # What works: stego.Y[:] = stego_Y[:]

        self.assertTrue(np.array_equal(stego_Y, stego.Y))
        self.assertFalse(np.array_equal(cover.Y, stego.Y))

        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            stego.write_dct(f.name)

        self.assertTrue(np.array_equal(stego_Y, stego.Y))
        self.assertFalse(np.array_equal(cover.Y, stego.Y))


__all__ = ["TestUERD"]
