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
        rhoP1 = np.array([50]*n)
        rhoM1 = rhoP1.copy()
        # compute probability
        (pP1, pM1), lbda = cl.simulate._ternary.probability(
            rhoP1, rhoM1,
            alpha,
            len(rhoP1),
        )
        # estimate average relative payload
        _, Hp = cl.simulate.average_payload(lbda=lbda, pP1=pP1, pM1=pM1)
        alpha_hat = Hp/n
        self.assertAlmostEqual(alpha, alpha_hat, 3)

    @parameterized.expand([[.05], [.1], [.2], [.4]])
    def test_simulate_ternary_random(self, alpha):
        self._logger.info('TestSimulate.test_simulate_ternary_random')
        #
        n = int(1000)
        rng = np.random.default_rng(12345)
        rhoP1 = rng.normal(.1, .05, n)  # 1000 samples ~ N(.1, .05)
        rhoP1[rhoP1 < 0] = 0
        rhoM1 = rhoP1.copy()
        (pP1, pM1), lbda = cl.simulate._ternary.probability(
            rhoP1, rhoM1,
            alpha,
            len(rhoP1),
        )
        # estimate average relative payload
        _, Hp = cl.simulate.average_payload(lbda=lbda, pP1=pP1, pM1=pM1)
        alpha_hat = Hp/n
        self.assertAlmostEqual(alpha, alpha_hat, 3)


__all__ = ['TestSimulate']
