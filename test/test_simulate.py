"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import tempfile
import time
import unittest
from unittest import mock

from . import defs


class TestSimulate(unittest.TestCase):
    """Test suite for simulate module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([
        [alpha, solver]
        for alpha in [.05, .1, .2, .4, 1.2]
        for solver in ['SOLVER_BSEARCH', 'SOLVER_BSEARCH_DDE']
    ])
    def test_simulate_ternary_constant(self, alpha, solver):
        self._logger.info(f'TestSimulate.test_simulate_ternary_constant({alpha=}, {solver=})')
        # constant distortion
        n = 1000
        rho_p1 = np.array([50]*n)
        rho_m1 = rho_p1.copy()
        # compute lambda
        lbda = cl.simulate.search(
            target=int(np.round(alpha * n)),
            rhos=(rho_p1, rho_m1),
            n=n,
            solver=getattr(cl.simulate, solver),
        )
        # estimate average relative payload
        Hp = cl.simulate.entropy(lbda=lbda, rhos=(rho_p1, rho_m1))
        alpha_hat = Hp / n
        np.testing.assert_allclose(alpha, alpha_hat, atol=.03)

    @parameterized.expand([
        [alpha, solver]
        for alpha in [.05, .1, .2, .4, 1.2]
        for solver in ['SOLVER_BSEARCH', 'SOLVER_BSEARCH_DDE']
    ])
    def test_simulate_ternary_random(self, alpha, solver):
        self._logger.info(f'TestSimulate.test_simulate_ternary_random({alpha=}, {solver=})')
        #
        n = int(1000)
        rng = np.random.default_rng(12345)
        rho_p1 = rng.normal(.1, .05, n)  # 1000 samples ~ N(.1, .05)
        rho_p1[rho_p1 < 0] = 0
        rho_m1 = rho_p1.copy()
        lbda = cl.simulate.search(
            target=int(np.round(alpha * n)),
            rhos=(rho_p1, rho_m1),
            n=n,
            solver=getattr(cl.simulate, solver),
        )
        # estimate average relative payload
        Hp = cl.simulate.entropy(lbda=lbda, rhos=(rho_p1, rho_m1))
        alpha_hat = Hp/n
        np.testing.assert_allclose(alpha, alpha_hat, atol=.03)

    @parameterized.expand([
        [cost, solver]
        for cost in [1e-5, 1e-4, 1e-3, 1e-2]
        for solver in ['SOLVER_BSEARCH', 'SOLVER_BSEARCH_DDE']
    ])
    def test_simulate_ternary_dls(self, cost, solver):
        self._logger.info(f'TestSimulate.test_simulate_ternary_dls({cost=}, {solver=})')
        #
        n = int(10000)
        rng = np.random.default_rng(12345)
        rho_p1 = rng.normal(.1, .05, n)  # 1000 samples ~ N(.1, .05)
        rho_p1[rho_p1 < 0] = 0
        rho_m1 = rho_p1.copy()
        lbda = cl.simulate.search(
            target=cost * n,
            rhos=(rho_p1, rho_m1),
            n=n,
            objective=cl.DiLS,
            solver=getattr(cl.simulate, solver),
        )
        # estimate average distortion
        cost_hat = cl.simulate.distortion(lbda=lbda, rhos=(rho_p1, rho_m1))
        cost_hat = cost_hat / n
        np.testing.assert_allclose(cost, cost_hat, atol=.1)

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



class TestSimulator(unittest.TestCase):
    """Test suite for the simulator."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        rng = np.random.default_rng(12345)
        self.rhos = np.abs(rng.normal(0, 1., size=(2, 64, 48)))
        self.ps = rng.uniform(0, .4, size=(2, 64, 48))
        self.ps[:, ::7] = 0  # no change possible

    @parameterized.expand([
        [generator, order, seed]
        for generator in [None, 'MT19937']
        for order in ['C', 'F']
        for seed in [1, 12345]
    ])
    def test_sample_legacy(self, generator, order, seed):
        self._logger.info(f'TestSimulator.test_sample_legacy({generator}, {order}, {seed})')
        # ternary and binary, bit-exact with the legacy samplers
        for ps, legacy in [
            (self.ps, cl.simulate_old._ternary.simulate),
            (self.ps[:1], cl.simulate_old._binary.simulate),
        ]:
            delta = cl.simulate.sample(ps, generator=generator, order=order, seed=seed)
            delta_ref = legacy(ps=tuple(ps), generator=generator, order=order, seed=seed)
            np.testing.assert_array_equal(delta, delta_ref)

    def test_sample_frequency(self):
        self._logger.info('TestSimulator.test_sample_frequency')
        # q-ary changes, frequencies follow probabilities
        ps = np.stack([np.full(100_000, p) for p in (.1, .2, .05, .15)])
        values = (1, -1, 2, -2)
        delta = cl.simulate.sample(ps, values=values, seed=12345)
        for p, v in zip(ps[:, 0], values):
            np.testing.assert_allclose(np.mean(delta == v), p, atol=.005)
        np.testing.assert_allclose(np.mean(delta == 0), 1 - ps[:, 0].sum(), atol=.005)

    def test_sample_values(self):
        self._logger.info('TestSimulator.test_sample_values')
        # default values for one and two changes
        self.assertTrue(set(np.unique(cl.simulate.sample(self.ps[:1], seed=1))) <= {0, 1})
        self.assertTrue(set(np.unique(cl.simulate.sample(self.ps, seed=1))) <= {-1, 0, 1})
        # more changes need values
        ps = np.concatenate([self.ps, self.ps]) / 2
        with self.assertRaises(ValueError):
            cl.simulate.sample(ps, seed=1)
        with self.assertRaises(ValueError):
            cl.simulate.sample(ps, values=(1, -1), seed=1)
        with self.assertRaises(NotImplementedError):
            cl.simulate.sample(self.ps, generator='unknown', seed=1)

    def test_sample_explicit_no_change(self):
        self._logger.info('TestSimulator.test_sample_explicit_no_change')
        n = 200_000
        rhos = np.abs(np.random.default_rng(12345).normal(0, 1., size=(2, n)))
        # implicit no-change state
        ps = cl.simulate.get_probability(rhos, 1.)
        # explicit no-change state, as the first state
        rhos_zero = np.concatenate([np.zeros((1, n)), rhos])
        ps_zero = cl.simulate.get_probability(rhos_zero, 1., add_zero=False)
        delta = cl.simulate.sample(ps_zero, values=(0, 1, -1), seed=12345)
        for p, v in zip(ps, (1, -1)):
            np.testing.assert_allclose(np.mean(delta == v), p.mean(), atol=.005)
        # binary with explicit no-change state needs values
        ps_zero = cl.simulate.get_probability(rhos_zero[:2], 1., add_zero=False)
        delta = cl.simulate.sample(ps_zero, values=(0, 1), seed=12345)
        np.testing.assert_allclose(np.mean(delta == 1), ps_zero[1].mean(), atol=.005)
        # without values, both are changes (1, -1), no element is kept
        delta = cl.simulate.sample(ps_zero, seed=12345)
        self.assertTrue(np.all(delta != 0))
        # empty input
        self.assertEqual(cl.simulate.sample(np.empty((2, 0)), seed=12345).size, 0)

    def test_sample_invalid_probabilities(self):
        self._logger.info('TestSimulator.test_sample_invalid_probabilities')
        # probabilities over 1
        with self.assertRaises(ValueError):
            cl.simulate.sample(self.ps + .5, seed=12345)
        with self.assertRaises(ValueError):
            cl.simulate.sample(np.stack([self.ps[0], 1 - self.ps[0] + .1]), values=(0, 1), seed=12345)
        # explicit no-change state not summing to 1
        with self.assertWarns(RuntimeWarning):
            cl.simulate.sample(self.ps, values=(0, 1), seed=12345)
        # rounding error within tolerance
        ps = np.stack([self.ps[0], 1 - self.ps[0] + 1e-12])
        cl.simulate.sample(ps, values=(0, 1), seed=12345)

    def test_sample_legacy_no_change(self):
        self._logger.info('TestSimulator.test_sample_legacy_no_change')
        # legacy samplers took the no-change state last, and ignored it
        p0 = 1 - self.ps.sum(axis=0)
        with self.assertWarns(DeprecationWarning):
            delta = cl.simulate._ternary.simulate((*self.ps, p0), seed=12345)
        np.testing.assert_array_equal(delta, cl.simulate.sample(self.ps, seed=12345))
        with self.assertWarns(DeprecationWarning):
            delta = cl.simulate._binary.simulate((self.ps[0], 1 - self.ps[0]), seed=12345)
        np.testing.assert_array_equal(delta, cl.simulate.sample(self.ps[:1], seed=12345))

    @parameterized.expand([
        [solver, alpha]
        for solver in ['SOLVER_BSEARCH_DDE', 'SOLVER_NEWTON', 'SOLVER_BRENT']
        for alpha in [.05, .2, .4]
    ])
    def test_probability_pls(self, solver, alpha):
        self._logger.info(f'TestSimulator.test_probability_pls({solver}, {alpha})')
        n = self.rhos[0].size
        ps, lbda = cl.simulate.probability(self.rhos, alpha, solver=getattr(cl.simulate, solver))
        self.assertEqual(ps.shape, self.rhos.shape)
        np.testing.assert_array_equal(ps, cl.simulate.get_probability(self.rhos, lbda))
        # payload of the probabilities
        np.testing.assert_allclose(cl.simulate.entropy(self.rhos, ps=ps) / n, alpha, atol=1e-3)

    @parameterized.expand([
        ['SOLVER_BSEARCH_DDE', 'PLS', 307],  # whole bits, as in DDE
        ['SOLVER_NEWTON', 'PLS', 307.2],
        ['SOLVER_BRENT', 'PLS', 307.2],
        ['SOLVER_BSEARCH_DDE', 'DLS', 307.2],  # distortion is not rounded
    ])
    def test_probability_target(self, solver, sender, target):
        self._logger.info(f'TestSimulator.test_probability_target({solver}, {sender})')
        n = self.rhos[0].size  # 3072, i.e., alpha = .1 is 307.2
        with mock.patch.object(cl.simulate, 'search', return_value=1.) as search:
            cl.simulate.probability(
                self.rhos, .1,
                objective=getattr(cl.simulate, sender),
                solver=getattr(cl.simulate, solver),
            )
        self.assertAlmostEqual(search.call_args.kwargs['target'], target)

    @parameterized.expand([['SOLVER_BSEARCH'], ['SOLVER_NEWTON'], ['SOLVER_BRENT']])
    def test_probability_dils(self, solver):
        self._logger.info(f'TestSimulator.test_probability_dils({solver})')
        n = self.rhos[0].size
        cost = .05  # per element
        ps, _ = cl.simulate.probability(
            self.rhos, cost,
            objective=cl.simulate.DLS,
            solver=getattr(cl.simulate, solver),
            tol=1e-3 * cost * n,
        )
        np.testing.assert_allclose(cl.simulate.distortion(self.rhos, ps=ps) / n, cost, rtol=1e-3)

    def test_simulate(self):
        self._logger.info('TestSimulator.test_simulate')
        # simulate = probability + sample
        ps, _ = cl.simulate.probability(self.rhos, .4)
        delta = cl.simulate.simulate(self.rhos, .4, seed=12345)
        np.testing.assert_array_equal(delta, cl.simulate.sample(ps, seed=12345))
        # legacy positional order (rhos, alpha, n, seed)
        delta = cl.simulate.simulate(self.rhos, .4, self.rhos[0].size, 12345)
        np.testing.assert_array_equal(delta, cl.simulate.sample(ps, seed=12345))
        # binary
        ps, _ = cl.simulate.probability(self.rhos[:1], .4)
        delta = cl.simulate.simulate(self.rhos[:1], .4, seed=12345)
        np.testing.assert_array_equal(delta, cl.simulate.sample(ps, seed=12345))


class TestLegacyInterface(unittest.TestCase):
    """Test suite for the deprecated interface of conseal.simulate."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        rng = np.random.default_rng(12345)
        self.ps = tuple(rng.uniform(0, .3, size=(2, 1000)))
        self.rhos = tuple(np.abs(rng.normal(0, 1., size=(2, 1000))))

    @parameterized.expand([
        ['_ternary.simulate'],
        ['_binary.simulate'],
    ])
    def test_deprecated_simulate(self, name):
        self._logger.info(f'TestLegacyInterface.test_deprecated_simulate({name})')
        module, func = name.split('.')
        ps = self.ps if module == '_ternary' else self.ps[:1]
        with self.assertWarns(DeprecationWarning):
            delta = getattr(getattr(cl.simulate, module), func)(ps=ps, seed=12345)
        # delegates to the legacy implementation
        delta_old = getattr(getattr(cl.simulate_old, module), func)(ps=ps, seed=12345)
        np.testing.assert_array_equal(delta, delta_old)

    def test_deprecated_get_p(self):
        self._logger.info('TestLegacyInterface.test_deprecated_get_p')
        with self.assertWarns(DeprecationWarning):
            p = cl.simulate.get_p(1., *self.rhos)
        np.testing.assert_array_equal(p, cl.simulate_old.get_p(1., *self.rhos))

    @parameterized.expand([
        [name] for name in [
            'binary', 'ternary', 'get_p', 'average_payload', 'average_distortion',
            '_binary.probability', '_binary.simulate', '_binary.binary',
            '_ternary.probability', '_ternary.simulate', '_ternary.ternary',
        ]
    ])
    def test_deprecated_interface(self, name):
        self._logger.info(f'TestLegacyInterface.test_deprecated_interface({name})')
        # same name and documentation as the legacy implementation
        new, old = cl.simulate, cl.simulate_old
        for part in name.split('.'):
            new, old = getattr(new, part), getattr(old, part)
        self.assertEqual(new.__doc__, old.__doc__)
        # samplers are forwarded to the replacement, the rest to the legacy implementation
        if name in ('_binary.simulate', '_ternary.simulate'):
            self.assertIsNot(new.__wrapped__, old)
        else:
            self.assertIs(new.__wrapped__, old)


__all__ = ['TestSimulate', 'TestSimulator', 'TestLegacyInterface']