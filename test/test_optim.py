"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import tempfile
import time
import unittest

from . import defs


class TestOptim(unittest.TestCase):
    """Test suite for new optimization module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    # @parameterized.expand([[.1]])
    @parameterized.expand([[.05], [.1], [.2], [.4]])
    def test_ternary_normal(self, alpha):
        self._logger.info(f'TestOptim.test_ternary_normal({alpha})')
        #
        n = 1_000
        target = alpha * n
        # draw costs rho ~ N(0,1)
        rho_p1, rho_m1 = np.abs(self.rng.normal(loc=0, scale=10., size=(2, n)))  # homoscedastic costs
        rhos = rho_p1, rho_m1
        # binary search for PLS
        lbda1 = cl.simulate.optim_new.search(
            target=target, rhos=rhos,
            sender=cl.simulate.PAYLOAD_LIMITED_SENDER,
            solver=cl.simulate.SOLVER_BSEARCH,
            tol=1,
        )
        f_val = cl.simulate.optim_new.entropy(rhos=rhos, lbda=lbda1)
        np.testing.assert_allclose(target, f_val, atol=1)
        # Newton for PLS
        lbda1 = cl.simulate.optim_new.search(
            target=target, rhos=rhos,
            sender=cl.simulate.PAYLOAD_LIMITED_SENDER,
            solver=cl.simulate.SOLVER_NEWTON,
            tol=1,
        )
        f_val = cl.simulate.optim_new.entropy(rhos=rhos, lbda=lbda1)
        np.testing.assert_allclose(target, f_val, atol=1)

        # get expected distrotion
        target = cl.simulate.optim_new.distortion(rhos=rhos, lbda=lbda1)
        print(lbda1, f_val / n, target / n)
        # binary search for DiLS
        lbda1 = cl.simulate.optim_new.search(
            target=target, rhos=rhos,
            sender=cl.simulate.DISTORTION_LIMITED_SENDER,
            solver=cl.simulate.SOLVER_BSEARCH,
            tol=1,
        )
        f_val = cl.simulate.optim_new.distortion(rhos=rhos, lbda=lbda1)
        print(lbda1, f_val / n)
        np.testing.assert_allclose(target, f_val, atol=1)
        # # Newton for DiLS
        # lbda1 = cl.simulate.optim_new.search(
        #     target=target, rhos=rhos,
        #     sender=cl.simulate.DISTORTION_LIMITED_SENDER,
        #     solver=cl.simulate.SOLVER_NEWTON,
        #     tol=1,
        # )
        # f_val = cl.simulate.optim_new.distortion(rhos=rhos, lbda=lbda1)
        # np.testing.assert_allclose(target, f_val, atol=1)



    # @parameterized.expand([[.1]])  # [.05], [.1], [.2], [.4]])
    # def test_ternary_normal_qary(self, alpha):
    #     self._logger.info(f'TestOptim.test_ternary_normal_qary({alpha})')
    #     #
    #     n = 10_000
    #     target = alpha * n
    #     # draw costs rho ~ N(0,1)
    #     rhos = np.abs(self.rng.normal(loc=0, scale=10., size=(8, n)))  # homoscedastic costs
    #     # binary search for PLS
    #     lbda1 = cl.simulate.optim_new.search(
    #         target=target, rhos=rhos,
    #         sender=cl.simulate.PAYLOAD_LIMITED_SENDER,
    #         solver=cl.simulate.SOLVER_BSEARCH,
    #         tol=1, q=8,
    #     )
    #     f_val = cl.simulate.optim_new.entropy(rhos=rhos, lbda=lbda1, q=8)
    #     np.testing.assert_allclose(target, f_val, atol=1)
    #     # Newton for PLS
    #     lbda1 = cl.simulate.optim_new.search(
    #         target=target, rhos=rhos,
    #         sender=cl.simulate.PAYLOAD_LIMITED_SENDER,
    #         solver=cl.simulate.SOLVER_NEWTON,
    #         tol=1, q=8,
    #     )
    #     f_val = cl.simulate.optim_new.entropy(rhos=rhos, lbda=lbda1, q=8)
    #     np.testing.assert_allclose(target, f_val, atol=1)

    #     # get expected distrotion
    #     target = cl.simulate.optim_new.distortion(rhos=rhos, lbda=lbda1)
    #     # binary search for DiLS
    #     lbda1 = cl.simulate.optim_new.search(
    #         target=target, rhos=rhos,
    #         sender=cl.simulate.DISTORTION_LIMITED_SENDER,
    #         solver=cl.simulate.SOLVER_BSEARCH,
    #         tol=1, q=8,
    #     )
    #     f_val = cl.simulate.optim_new.distortion(rhos=rhos, lbda=lbda1, q=8)
    #     np.testing.assert_allclose(target, f_val, atol=1)
    #     # Newton for DiLS
    #     lbda1 = cl.simulate.optim_new.search(
    #         target=target, rhos=rhos,
    #         sender=cl.simulate.DISTORTION_LIMITED_SENDER,
    #         solver=cl.simulate.SOLVER_NEWTON,
    #         tol=1, q=8,
    #     )
    #     f_val = cl.simulate.optim_new.distortion(rhos=rhos, lbda=lbda1, q=8)
    #     np.testing.assert_allclose(target, f_val, atol=1)





    # @parameterized.expand([[.05], [.1], [.2], [.4]])
    # def test_simulate_ternary_random(self, alpha):
    #     self._logger.info('TestSimulate.test_simulate_ternary_random')
    #     #
    #     n = int(1000)
    #     rng = np.random.default_rng(12345)
    #     rho_p1 = rng.normal(.1, .05, n)  # 1000 samples ~ N(.1, .05)
    #     rho_p1[rho_p1 < 0] = 0
    #     rho_m1 = rho_p1.copy()
    #     ps, lbda = cl.simulate._ternary.probability(
    #         rhos=(rho_p1, rho_m1),
    #         alpha=alpha,
    #         n=len(rho_p1),
    #     )
    #     # estimate average relative payload
    #     Hp = cl.simulate.entropy(lbda=lbda, rhos=(rho_p1, rho_m1))
    #     alpha_hat = Hp/n
    #     # self.assertAlmostEqual(alpha, alpha_hat, 1)
    #     np.testing.assert_allclose(alpha, alpha_hat, atol=.03)

    # @parameterized.expand([[1e-5], [1e-4], [1e-3], [1e-2]])
    # def test_simulate_ternary_dls(self, cost):
    #     self._logger.info('TestSimulate.test_simulate_ternary_dls')
    #     #
    #     n = int(10000)
    #     rng = np.random.default_rng(12345)
    #     rho_p1 = rng.normal(.1, .05, n)  # 1000 samples ~ N(.1, .05)
    #     rho_p1[rho_p1 < 0] = 0
    #     rho_m1 = rho_p1.copy()
    #     ps, lbda = cl.simulate._ternary.probability(
    #         rhos=(rho_p1, rho_m1),
    #         alpha=cost,
    #         n=n,
    #         sender=cl.DiLS
    #     )
    #     # #
    #     # _, H = cl.simulate.average_payload(lbda=lbda, ps=ps)
    #     # print(cost, H / n)
    #     # estimate average distortion
    #     # _, cost_hat = cl.simulate.expected_distortion(rhos=(rho_p1, rho_m1), ps=ps)
    #     # cost_hat = cost_hat / n
    #     # self.assertAlmostEqual(cost, cost_hat, 3)
    #     #
    #     cost_hat = cl.simulate.expected_distortion(lbda=lbda, rhos=(rho_p1, rho_m1))
    #     cost_hat = cost_hat / n
    #     # self.assertAlmostEqual(cost, cost_hat, 3)
    #     np.testing.assert_allclose(cost, cost_hat, atol=.1)

    # @parameterized.expand([[.01], [.05], [.1], [.2], [.4], [1.]])
    # def test_simulate_pls_newton(self, alpha):
    #     self._logger.info(f'TestSimulate.test_simulate_pls_newton({alpha=})')
    #     #
    #     n = int(1e6)
    #     rng = np.random.default_rng(12345)
    #     rho_pm1 = np.clip(rng.normal(.15, .05, n), 1e-3, None)   # 1000 samples ~ N(.15, .05)
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._binary.probability(
    #         # rhos=(rho_p1, rho_m1),
    #         rhos=(rho_pm1,),
    #         alpha=alpha,
    #         n=n,
    #         sender=cl.PLS,
    #         lambda_optimizer=cl.simulate.NEWTON,
    #         # lbda0=(1000,)
    #     )
    #     end = time.perf_counter()
    #     # estimate embedding rate
    #     h_hat = cl.simulate.entropy(rhos=(rho_pm1,), lbda=lbda)
    #     alpha_hat = h_hat / n
    #     duration = (end - start) * 1000
    #     print(f'Newton: {alpha=} {alpha_hat=} | {duration:.4f} ms')
    #     self.assertAlmostEqual(alpha, alpha_hat, 2)

    # @parameterized.expand([[.01], [.05]])
    # def test_simulate_dils_newton(self, cost: float):
    #     self._logger.info(f'TestSimulate.test_simulate_dils_newton({cost=})')
    #     #
    #     n = int(1e6)
    #     rng = np.random.default_rng(12345)
    #     rho_p1 = np.clip(rng.normal(.15, .15, n), 1e-10, 10**10)   # 1000 samples ~ N(.1, .05)
    #     # rho_p1 = rng.normal(.1, .05, n)  # 1000 samples ~ N(.1, .05)
    #     # rho_p1[rho_p1 < 0] = 0
    #     rho_m1 = rho_p1.copy()
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._ternary.probability(
    #         rhos=(rho_p1, rho_m1),
    #         # rhos=(rho_pm1,),
    #         alpha=cost,
    #         n=n,
    #         sender=cl.DiLS,
    #         lambda_optimizer=cl.simulate.NEWTON,
    #         # lbda0=(10,)
    #         lbda0=(np.sqrt(n),),
    #     )
    #     end = time.perf_counter()
    #     # estimate average distortion
    #     Erho_hat = cl.simulate.expected_distortion(rhos=(rho_p1, rho_m1), lbda=lbda)
    #     cost_hat = Erho_hat / n
    #     duration = (end - start) * 1000
    #     print(f'Newton (DiLS): {cost=} {cost_hat=} | {duration:.4f} ms')
    #     self.assertAlmostEqual(cost, cost_hat, 3)

    # @parameterized.expand([[.01], [.05], [.1], [.2], [.4], [1.]])
    # def test_simulate_pls_taylor_newton(self, alpha):
    #     self._logger.info(f'TestSimulate.test_simulate_pls_taylor_newton({alpha=})')
    #     #
    #     n = int(1e6)
    #     rng = np.random.default_rng(12345)
    #     rho_pm1 = np.clip(rng.normal(10., .1, n), 1e-10, 10**10)   # 1000 samples ~ N(.15, .05)
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._binary.probability(
    #         # rhos=(rho_p1, rho_m1),
    #         rhos=(rho_pm1,),
    #         alpha=alpha,
    #         n=n,
    #         sender=cl.PLS,
    #         lambda_optimizer=cl.simulate.NEWTON,
    #         max_iter=50,
    #         # lbda0=(5,),
    #     )
    #     end = time.perf_counter()
    #     # estimate embedding rate
    #     h_hat = cl.simulate.entropy(rhos=(rho_pm1,), lbda=lbda)
    #     alpha_hat = h_hat / n
    #     duration = (end - start) * 1000
    #     # print(f'{alpha=} {alpha_hat=}')
    #     print(f'Taylor-Newton: {alpha=} {alpha_hat=} | {duration:.4f} ms')
    #     self.assertAlmostEqual(alpha, alpha_hat, 2)

    # @parameterized.expand([
    #     [f, alpha]
    #     for alpha in [.1, .2, .4]
    #     for f in defs.TEST_IMAGES[:1]
    # ])
    # def test_simulate_newton(self, fname: str, alpha: float):
    #     self._logger.info(f'TestSimulate.test_simulate_newton({fname=}, {alpha=})')
    #     # load cover
    #     x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
    #     n = x.size
    #     # print(n)
    #     # simulate the stego
    #     # alpha = .4
    #     rho_pm1 = cl.hill._costmap.compute_cost(x)
    #     rhos = (rho_pm1,)
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._binary.probability(
    #         rhos=rhos,
    #         alpha=alpha,
    #         n=n,
    #         lambda_optimizer=cl.simulate.NEWTON,
    #         sender=cl.PAYLOAD_LIMITED_SENDER,
    #         # lbda0=(24,),
    #         # max_iter=100,
    #     )
    #     end = time.perf_counter()
    #     # estimate embedding rate
    #     h_hat = cl.simulate.entropy(rhos=rhos, lbda=lbda)
    #     alpha_hat = h_hat / n
    #     duration = (end - start) * 1000
    #     # alpha_hat = cl.tools.entropy(*ps) / n
    #     print(f'DDE binary search: {alpha=} {alpha_hat=} | {duration:.4f} ms')
    #     self.assertAlmostEqual(alpha, alpha_hat, 2)



    # @parameterized.expand([[.01], [.05], [.1], [.2], [.4], [1.]])
    # def test_simulate_pls_polynomial_proxy(self, alpha):
    #     self._logger.info(f'TestSimulate.test_simulate_pls_polynomial_proxy({alpha=})')
    #     #
    #     n = int(1e6)
    #     rng = np.random.default_rng(12345)
    #     rho_pm1 = np.clip(rng.normal(.15, .05, n), 1e-3, None)   # 1000 samples ~ N(.15, .05)
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._binary.probability(
    #         # rhos=(rho_p1, rho_m1),
    #         rhos=(rho_pm1,),
    #         alpha=alpha,
    #         n=n,
    #         sender=cl.PLS,
    #         lbda0=(5, 20, 45),
    #         lambda_optimizer=cl.simulate.POLYNOMIAL_PROXY
    #     )
    #     end = time.perf_counter()
    #     # estimate embedding rate
    #     h_hat = cl.simulate.entropy(rhos=(rho_pm1,), lbda=lbda)
    #     alpha_hat = h_hat / n
    #     duration = (end - start) * 1000
    #     print(f'Polynomial proxy: {alpha=} {alpha_hat=} | {duration:.4f} ms')
    #     self.assertAlmostEqual(alpha, alpha_hat, 1)


    # @parameterized.expand([
    #     [f, alpha]
    #     for alpha in [.1, .2, .4]
    #     for f in defs.TEST_IMAGES[:1]
    # ])
    # def test_simulate_binary_search_dde(self, fname: str, alpha: float):
    #     self._logger.info(f'TestSimulate.test_simulate_binary_search_dde({fname=}, {alpha=})')
    #     # load cover
    #     x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
    #     n = x.size
    #     # print(n)
    #     # simulate the stego
    #     # alpha = .4
    #     rhos = cl.hill.compute_cost_adjusted(x)
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._ternary.probability(
    #         rhos=rhos,
    #         alpha=alpha,
    #         n=n,
    #         lambda_optimizer=cl.simulate.BINARY_SEARCH_DDE,
    #         sender=cl.PAYLOAD_LIMITED_SENDER,
    #         # lbda0=(10,),
    #         # max_iter=100,
    #     )
    #     end = time.perf_counter()
    #     # estimate embedding rate
    #     h_hat = cl.simulate.entropy(rhos=rhos, lbda=lbda)
    #     alpha_hat = h_hat / n
    #     duration = (end - start) * 1000
    #     # alpha_hat = cl.tools.entropy(*ps) / n
    #     print(f'DDE binary search: {alpha=} {alpha_hat=} | {duration:.4f} ms')
    #     self.assertAlmostEqual(alpha, alpha_hat, 2)










    # @parameterized.expand([[.01], [.05], [.1], [.2], [.4]])
    # def test_simulate_pls_taylor_inverse(self, alpha):
    #     self._logger.info(f'TestSimulate.test_simulate_pls_taylor_inverse({alpha=})')
    #     #
    #     n = int(1e6)
    #     rng = np.random.default_rng(12345)
    #     rho_pm1 = np.clip(rng.normal(.15, .05, n), 1e-3, None)   # 1000 samples ~ N(.15, .05)
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._binary.probability(
    #         rhos=(rho_pm1,),
    #         alpha=alpha,
    #         n=n,
    #         sender=cl.PLS,
    #         lbda0=(24,),
    #         lambda_optimizer=cl.simulate.TAYLOR_INVERSE
    #     )
    #     end = time.perf_counter()
    #     # estimate embedding rate
    #     h_hat = cl.simulate.entropy(rhos=(rho_pm1,), lbda=lbda)
    #     alpha_hat = h_hat / n
    #     duration = (end - start) * 1000
    #     print(f'Taylor: {alpha=} {alpha_hat=} | {duration:.4f} ms')
    #     np.testing.assert_allclose(alpha, alpha_hat, atol=.1)

    # @parameterized.expand([[.01], [.05], [.1], [.2], [.4], [1.]])
    # def test_simulate_pls_binary_search(self, alpha):
    #     self._logger.info(f'TestSimulate.test_simulate_pls_binary_search({alpha=})')
    #     #
    #     n = int(1e6)
    #     rng = np.random.default_rng(12345)
    #     rho_pm1 = np.clip(rng.normal(.15, .05, n), 1e-3, None)   # 1000 samples ~ N(.15, .05)
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._binary.probability(
    #         # rhos=(rho_p1, rho_m1),
    #         rhos=(rho_pm1,),
    #         alpha=alpha,
    #         n=n,
    #         sender=cl.PLS,
    #         lambda_optimizer=cl.simulate.BINARY_SEARCH,
    #         lbda0=(0, 1e4)
    #     )
    #     end = time.perf_counter()
    #     # estimate embedding rate
    #     h_hat = cl.simulate.entropy(rhos=(rho_pm1,), lbda=lbda)
    #     alpha_hat = h_hat / n
    #     duration = (end - start) * 1000
    #     # print(f'{alpha=} {alpha_hat=}')
    #     print(f'Binary search: {alpha=} {alpha_hat=} | {duration:.4f} ms')
    #     self.assertAlmostEqual(alpha, alpha_hat, 2)

    # @parameterized.expand([
    #     [f, alpha]
    #     for alpha in [.1, .2, .4]
    #     for f in defs.TEST_IMAGES
    # ])
    # def test_simulate_newton(self, fname: str, alpha: float):
    #     self._logger.info(f'TestSimulate.test_simulate_newton({fname=}, {alpha=})')
    #     # load cover
    #     x = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
    #     n = x.size
    #     # print(n)
    #     # simulate the stego
    #     # alpha = .4
    #     rhos = cl.hill.compute_cost_adjusted(x)
    #     start = time.perf_counter()
    #     ps, lbda = cl.simulate._ternary.probability(
    #         rhos=rhos,
    #         alpha=alpha,
    #         n=n,
    #         lambda_optimizer=cl.simulate.BINARY_SEARCH_NEWTON,
    #         sender=cl.PAYLOAD_LIMITED_SENDER,
    #         # lbda0=(5000,),
    #         max_iter=100,
    #     )
    #     end = time.perf_counter()
    #     # estimate embedding rate
    #     h_hat = cl.simulate.entropy(rhos=rhos, lbda=lbda)
    #     alpha_hat = h_hat / n
    #     duration = (end - start) * 1000
    #     # alpha_hat = cl.tools.entropy(*ps) / n
    #     print(f'Newton: {alpha=} {alpha_hat=} | {duration:.4f} ms')
    #     self.assertAlmostEqual(alpha, alpha_hat, 2)


__all__ = ['TestSimulate']