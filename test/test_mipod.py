import logging
import os
from pathlib import Path
from parameterized import parameterized
import sys
import unittest
import numpy as np
from PIL import Image
from scipy.io import loadmat

sys.path.append('.')
import conseal as cl


COVER_DIR = Path("test/assets/cover")
MIPOD_DIR = Path("test/assets/mipod")


class TestMiPOD(unittest.TestCase):
    """Test suite for MiPOD embedding."""
    _logger = logging.getLogger(__name__)

    @parameterized.expand([
        ("uncompressed_gray/seal1.png", "seal1_embedding_change_probs.mat"),
        ("uncompressed_gray/seal2.png", "seal2_embedding_change_probs.mat"),
        ("uncompressed_gray/seal3.png", "seal3_embedding_change_probs.mat"),
        ("uncompressed_gray/seal4.png", "seal4_embedding_change_probs.mat"),
        ("uncompressed_gray/seal5.png", "seal5_embedding_change_probs.mat"),
        ("uncompressed_gray/seal6.png", "seal6_embedding_change_probs.mat"),
        ("uncompressed_gray/seal7.png", "seal7_embedding_change_probs.mat"),
        ("uncompressed_gray/seal8.png", "seal8_embedding_change_probs.mat"),
    ])
    def test_compare_embedding_change_probability(self, cover_name, stego_name):
        x0 = np.array(Image.open(COVER_DIR / cover_name))
        ps, _ = cl.mipod.probability(x0=x0, alpha=0.4)
        ps_matlab = loadmat(MIPOD_DIR / stego_name)["pChange"]

        np.testing.assert_allclose(ps[0], ps_matlab)
        # np.testing.assert_allclose(ps[1], ps_matlab)

    @parameterized.expand([
        ("uncompressed_gray/seal1.png", 1, "stego_seal1_seed_1.png"),
        ("uncompressed_gray/seal2.png", 2, "stego_seal2_seed_2.png"),
        ("uncompressed_gray/seal3.png", 3, "stego_seal3_seed_3.png"),
        ("uncompressed_gray/seal4.png", 4, "stego_seal4_seed_4.png"),
        ("uncompressed_gray/seal5.png", 5, "stego_seal5_seed_5.png"),
        ("uncompressed_gray/seal6.png", 6, "stego_seal6_seed_6.png"),
        ("uncompressed_gray/seal7.png", 7, "stego_seal7_seed_7.png"),
        ("uncompressed_gray/seal8.png", 8, "stego_seal8_seed_8.png"),
    ])
    def test_compare_matlab(self, cover_name, seed, stego_name):
        x0 = np.array(Image.open(COVER_DIR / cover_name))

        x1 = cl.mipod.simulate_single_channel(
            x0=x0,
            alpha=0.4,
            seed=seed,
            implementation=cl.MiPOD_ORIGINAL,
        )

        # Read stego image pre-computed with the Matlab implementation
        x1_matlab = np.array(Image.open(MIPOD_DIR / stego_name))
        np.testing.assert_allclose(x1, x1_matlab)

    # def test_flat(self):
    #     x0 = np.zeros((512, 512), dtype='uint8') * 128
    #     # x0 = np.random.randint(0, 256, size=(512, 512), dtype='uint8')
    #     # x0 = np.ones((512, 512), dtype='uint8') * 128
    #     x1 = cl.mipod.simulate_single_channel(
    #         x0=x0,
    #         alpha=0.4,
    #         seed=12345,
    #     )
    #     # print((x0 != x1).mean())
    #     # print(np.unique(x0 - x1))
    #     # print(x1)


__all__ = ["TestMiPOD"]
