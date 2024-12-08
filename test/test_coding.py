
import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from parameterized import parameterized
import scipy.io
import scipy.stats
import tempfile
import unittest

from . import defs


class TestCoding(unittest.TestCase):
    """Test suite for coding module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    def test_generate_H(self):
        self._logger.info('TestCoding.test_generate_H()')

        # Parameters
        m = 500  # Number of rows
        n = 10000
        c = .1  # Constant parameter
        delta = .05  # Failure probability

        # Generate the parity-check matrix
        H = cl.coding.wpc.generate_H(m, n=n, delta=delta, c=c, seed=12345)
        # print("Generated Parity-Check Matrix:\n", H)
        p_empirical = np.histogram(np.sum(H, axis=0), bins=m-1, range=(1, m), density=True)[0]
        p_theoretical = cl.coding.wpc.soliton(m=m, c=c, delta=delta)
        np.testing.assert_allclose(p_empirical[:99], p_theoretical[:99], atol=1e-2)

    def test_soliton(self):
        self._logger.info('TestCoding.test_soliton()')
        # get Soliton probabilities
        m = 100
        p = cl.coding.wpc.soliton(m, robust=False)
        # draw from reference Soliton
        rng = np.random.default_rng(12345)
        x_ref = np.ceil(1 / rng.random(int(1e6)))
        x_ref[x_ref > m] = 1
        # compare
        p_ref, _ = np.histogram(x_ref, bins=m, range=(1, m+1), density=True)
        # print(p, p_ref)
        np.testing.assert_allclose(p, p_ref, atol=1e-3)

    # def test_soliton_robust(self):
    #     self._logger.info('TestCoding.test_soliton_robust()')
    #     raise NotImplementedError

    @parameterized.expand([[alpha] for alpha in [1e-3, .01, .05, .1, .2, .4]])
    def test_hamming_efficiency(self, alpha):
        self._logger.info('TestCoding.test_hamming_efficiency')
        e_hamming = cl.coding.hamming.efficiency(alpha)
        e_bound = cl.coding.efficiency(alpha)
        self.assertGreater(e_bound, e_hamming)


__all__ = ["TestCoding"]
