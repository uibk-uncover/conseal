import logging
import os
from pathlib import Path
from parameterized import parameterized
import sys
import unittest
import jpeglib
import numpy as np
from scipy.io import loadmat

sys.path.append('.')
import conseal as cl


COVER_DIR = Path("test/assets/cover/jpeg_75_gray")
JMIPOD_DIR = Path("test/assets/jmipod")


class TestJMiPOD(unittest.TestCase):
    """Test suite for J-MiPOD embedding."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([
        ("seal1.jpg", "seal1_costmap.mat"),
        ("seal2.jpg", "seal2_costmap.mat"),
        ("seal3.jpg", "seal3_costmap.mat"),
        ("seal4.jpg", "seal4_costmap.mat"),
        ("seal5.jpg", "seal5_costmap.mat"),
        ("seal6.jpg", "seal6_costmap.mat"),
        ("seal7.jpg", "seal7_costmap.mat"),
        ("seal8.jpg", "seal8_costmap.mat"),
    ])
    def test_compare_costmap_matlab(self, cover_name, costmap_name):
        cover_dct = jpeglib.read_dct(COVER_DIR / cover_name)
        costmap = cl.jmipod.compute_cost(y0=cover_dct.Y, qt=cover_dct.qt[0])
        costmap_matlab = loadmat(JMIPOD_DIR / costmap_name)["FisherInformation"]

        np.testing.assert_allclose(costmap, costmap_matlab)

    @parameterized.expand([
        ("seal1.jpg", 1, "stego_seal1_seed_1.jpg"),
        ("seal2.jpg", 2, "stego_seal2_seed_2.jpg"),
        ("seal3.jpg", 3, "stego_seal3_seed_3.jpg"),
        ("seal4.jpg", 4, "stego_seal4_seed_4.jpg"),
        ("seal5.jpg", 5, "stego_seal5_seed_5.jpg"),
        ("seal6.jpg", 6, "stego_seal6_seed_6.jpg"),
        ("seal7.jpg", 7, "stego_seal7_seed_7.jpg"),
        ("seal8.jpg", 8, "stego_seal8_seed_8.jpg"),
    ])
    def test_compare_matlab(self, cover_name, seed, stego_name):
        cover_dct = jpeglib.read_dct(COVER_DIR / cover_name)

        y1 = cl.jmipod.simulate_single_channel(
            y0=cover_dct.Y,
            qt=cover_dct.qt[0],
            alpha=0.4,
            seed=seed,
        )

        # Read stego DCT coefficients pre-computed with the Matlab implementation
        stego_dct_matlab = jpeglib.read_dct(JMIPOD_DIR / stego_name)
        np.testing.assert_allclose(y1, stego_dct_matlab.Y)


__all__ = ["TestJMiPOD"]
