
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sys
import tempfile
import unittest

from . import defs


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
    def test_distortion_hill(self, fname: str):
        self._logger.info(f'TestHILL.test_simulate_hill({fname})')
        # load cover
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        # simulate the stego
        alpha = .4
        rho_p1, rho_m1 = cl.hill.compute_cost_adjusted(x)
        (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
            rho_p1, rho_m1,
            alpha=alpha,
            n=x.size,
        )
        # estimate average relative payload
        _, Hp = cl.simulate.average_payload(lbda=lbda, p_p1=p_p1, p_m1=p_m1)
        alpha_hat = Hp/x.size
        self.assertAlmostEqual(alpha, alpha_hat, 3)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_simulate_hill(self, fname: str):
        self._logger.info(f'TestHILL.test_simulate_hill({fname})')
        # load cover
        x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))

        # simulate the stego
        alpha = .4
        y = cl.hill.simulate_single_channel(x, alpha, seed=12345)

        # check changes
        self.assertEqual(y.dtype, np.uint8)
        np.testing.assert_array_equal(x.shape, y.shape)
        d = x.astype('int16') - y.astype('int16')
        self.assertLessEqual(d.max(), 1)
        self.assertGreaterEqual(d.min(), -1)

