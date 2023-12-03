"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import jpeglib
import logging
import numpy as np
from parameterized import parameterized
import sys
import tempfile
import unittest

from defs import ASSETS_DIR, COVER_DIR
STEGO_DIR = ASSETS_DIR / 'nsF5'


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
        (
            "landscape_gray.jpeg",
            "landscape_gray_matlab_alpha_0.5_seed_12345.jpg",
            0.5,
            12345,
        ),
        (
            "lizard_gray.jpeg",
            "lizard_gray_matlab_alpha_0.4_seed_6789.jpg",
            0.4,
            6789,
        ),
        (
            "mountain_gray.jpeg",
            "mountain_gray_matlab_alpha_0.2_seed_6020.jpg",
            0.2,
            6020,
        ),
        (
            "nuclear_gray.jpeg",
            "nuclear_gray_matlab_alpha_0.1_seed_91058.jpg",
            0.1,
            91058,
        ),
    ])
    def test_matlab_equivalence(self, cover_filepath, matlab_stego_filepath, embedding_rate, seed):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
            cover_img = jpeglib.read_dct(COVER_DIR / cover_filepath)

            stego_Y = cl.nsF5._simulate.simulate(
                cover_dct_coeffs=cover_img.Y,
                embedding_rate=embedding_rate,
                seed=seed,
            )

            stego_img = cover_img
            stego_img.Y = stego_Y
            stego_img.write_dct(f.name)

            self.compare_JPEG_files(f.name, STEGO_DIR / matlab_stego_filepath)


__all__ = ["TestnsF5"]
