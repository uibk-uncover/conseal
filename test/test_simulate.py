"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
import tempfile
import unittest


class TestSimulate(unittest.TestCase):
    """Test suite for simulate module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[.05], [.1], [.2], [.4]])
    def test_simulate_ternary_constant(self, alpha):
        self._logger.info('TestSimulate.test_simulate_ternary_constant')
        # constant distortion
        n = 1000
        rho_p1 = np.array([50]*n)
        rho_m1 = rho_p1.copy()
        # compute probability
        ps, lbda = cl.simulate._ternary.probability(
            rhos=(rho_p1, rho_m1),
            alpha=alpha,
            n=len(rho_p1),
        )
        # estimate average relative payload
        _, Hp = cl.simulate.average_payload(lbda=lbda, ps=ps)
        alpha_hat = Hp/n
        self.assertAlmostEqual(alpha, alpha_hat, 3)

    @parameterized.expand([[.05], [.1], [.2], [.4]])
    def test_simulate_ternary_random(self, alpha):
        self._logger.info('TestSimulate.test_simulate_ternary_random')
        #
        n = int(1000)
        rng = np.random.default_rng(12345)
        rho_p1 = rng.normal(.1, .05, n)  # 1000 samples ~ N(.1, .05)
        rho_p1[rho_p1 < 0] = 0
        rho_m1 = rho_p1.copy()
        ps, lbda = cl.simulate._ternary.probability(
            rhos=(rho_p1, rho_m1),
            alpha=alpha,
            n=len(rho_p1),
        )
        # estimate average relative payload
        _, Hp = cl.simulate.average_payload(lbda=lbda, ps=ps)
        alpha_hat = Hp/n
        self.assertAlmostEqual(alpha, alpha_hat, 3)


__all__ = ['TestSimulate']
