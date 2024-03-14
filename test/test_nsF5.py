"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import conseal as cl
import jpeglib
import logging
import numpy as np
from parameterized import parameterized
import tempfile
import unittest
from . import defs


STEGO_DIR = defs.ASSETS_DIR / 'nsF5'


class TestnsF5(unittest.TestCase):
    """Test suite for nsF5 embedding."""
    _logger = logging.getLogger(__name__)

    def compare_JPEG_files(self, a, b):
        img_a = jpeglib.read_dct(a)
        img_b = jpeglib.read_dct(b)

        # Verify that both images have the same channels
        self.assertTrue((img_a.Cb is None) == (img_b.Cb is None))
        self.assertTrue((img_a.Cr is None) == (img_b.Cr is None))

        # Verify quantization table equivalence
        self.assertTrue(np.allclose(img_a.qt, img_b.qt))

        # Compare channels
        self.assertTrue(img_a.Y.shape == img_b.Y.shape)
        np.testing.assert_array_equal(img_a.Y, img_b.Y)
        self.assertTrue(np.allclose(img_a.Y, img_b.Y))

        if img_a.Cb:
            self.assertTrue(img_a.Cb.shape == img_b.Cb.shape)
            self.assertTrue(np.allclose(img_a.Cb, img_b.Cb))
        if img_a.Cr:
            self.assertTrue(img_a.Cr.shape == img_b.Cr.shape)
            self.assertTrue(np.allclose(img_a.Cr, img_b.Cr))

    @parameterized.expand([
        ("seal1.jpg", "seal1_alpha_0.4_seed_1.jpg", 0.4, 1),
        ("seal2.jpg", "seal2_alpha_0.4_seed_2.jpg", 0.4, 2),
        ("seal3.jpg", "seal3_alpha_0.4_seed_3.jpg", 0.4, 3),
        ("seal4.jpg", "seal4_alpha_0.4_seed_4.jpg", 0.4, 4),
        ("seal5.jpg", "seal5_alpha_0.4_seed_5.jpg", 0.4, 5),
        ("seal6.jpg", "seal6_alpha_0.4_seed_6.jpg", 0.4, 6),
        ("seal7.jpg", "seal7_alpha_0.4_seed_7.jpg", 0.4, 7),
        ("seal8.jpg", "seal8_alpha_0.4_seed_8.jpg", 0.4, 8),
    ])
    def test_matlab_equivalence(self, cover_filepath, matlab_stego_filepath, embedding_rate, seed):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            cover_img = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / cover_filepath)

            stego_Y = cl.nsF5.simulate_single_channel(
                cover_dct_coeffs=cover_img.Y,
                embedding_rate=embedding_rate,
                seed=seed,
            )

            stego_img = cover_img
            stego_img.Y = stego_Y
            stego_img.write_dct(f.name)

            self.compare_JPEG_files(f.name, STEGO_DIR / matlab_stego_filepath)


__all__ = ["TestnsF5"]
