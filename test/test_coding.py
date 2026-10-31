
import conseal as cl
from conseal.coding import _lt, _symbol
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

    @parameterized.expand([[2, 10, 200], [3, 6, 200], [5, 4, 100]])
    def test_generate_H(self, q, m, n):
        self._logger.info(f'TestCoding.test_generate_H({q=}, {m=}, {n=})')
        H = cl.coding.wpc.generate_H(m, n, q=q, seed=12345)
        self.assertEqual(H.shape, (m, n))
        self.assertTrue((H < q).all())
        # nonzero, normalized, and distinct columns
        leading = H[(H != 0).argmax(axis=0), np.arange(n)]
        np.testing.assert_array_equal(leading, 1)
        self.assertEqual(len(np.unique(H, axis=1).T), n)
        # reproducible
        np.testing.assert_array_equal(H, cl.coding.wpc.generate_H(m, n, q=q, seed=12345))

    @parameterized.expand([
        [q, p, n, backend]
        for q, p, n in [[2, 6, 12], [3, 3, 6], [5, 2, 4]]
        for backend in [cl.tools.BACKEND_PYTHON, cl.tools.BACKEND_RUST]
    ])
    def test_lt_solve(self, q, p, n, backend):
        self._logger.info(f'TestCoding.test_lt_solve({q=}, {p=}, {n=}, {backend=})')
        # brute force over all coefficient vectors of n random columns
        rng = np.random.default_rng(12345)
        H = rng.integers(0, q, (p, n))
        v_all = (np.arange(q**n)[:, None] // q**np.arange(n)) % q
        z_all = (H @ v_all.T % q).T
        weights = (v_all != 0).sum(axis=1)
        for z in np.unique(z_all, axis=0):
            with backend:
                v = _lt.lt_solve(H, z, q=q)
            np.testing.assert_array_equal(H @ v % q, z)
            # fewest nonzeros
            self.assertEqual((v != 0).sum(), weights[(z_all == z).all(axis=1)].min())

    @parameterized.expand([[2, 18, 45], [2, 14, 20], [3, 11, 30], [3, 8, 12], [5, 7, 20]])
    def test_lt_backends(self, q, p, n):
        self._logger.info(f'TestCoding.test_lt_backends({q=}, {p=}, {n=})')
        # same solution of random systems with both backends
        rng = np.random.default_rng(12345)
        for _ in range(20):
            H = rng.integers(0, q, (p, n), dtype='uint8')
            z = rng.integers(0, q, p, dtype='uint8')
            try:
                with cl.tools.BACKEND_PYTHON:
                    v_py = _lt.lt_solve(H, z, q=q)
            except ValueError:
                with cl.tools.BACKEND_RUST, self.assertRaises(ValueError):
                    _lt.lt_solve(H, z, q=q)
                continue
            with cl.tools.BACKEND_RUST:
                v_rs = _lt.lt_solve(H, z, q=q)
            np.testing.assert_array_equal(v_rs, v_py)

    @parameterized.expand([
        [q, backend]
        for q in [2, 3]
        for backend in [cl.tools.BACKEND_PYTHON, cl.tools.BACKEND_RUST]
    ])
    def test_lt_no_solution(self, q, backend):
        self._logger.info(f'TestCoding.test_lt_no_solution({q=})')
        # second row is sum of first and third, but z is not
        D = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
        with backend, self.assertRaises(ValueError):
            _lt.lt_solve(D, np.array([1, 1, 1]), q=q)

    @parameterized.expand([[4], [6]])
    def test_lt_not_prime(self, q):
        self._logger.info(f'TestCoding.test_lt_not_prime({q=})')
        with self.assertRaises(ValueError):
            _lt.lt_solve(np.eye(3, dtype=int), np.ones(3, dtype=int), q=q)

    @parameterized.expand([[q, seed] for q in [2, 3] for seed in range(3)])
    def test_symbols(self, q, seed):
        self._logger.info(f'TestCoding.test_symbols({q=}, {seed=})')
        message = np.random.default_rng(seed).bytes(37)
        symbols = _symbol.bytes_to_symbols(message, q)
        self.assertEqual(len(symbols), _symbol.num_symbols(37, q))
        self.assertTrue(((symbols >= 0) & (symbols < q)).all())
        self.assertEqual(_symbol.symbols_to_bytes(symbols, 37, q), message)

    @parameterized.expand([[q, seed] for q in [2, 3, 5] for seed in range(3)])
    def test_wpc_embed_extract(self, q, seed):
        self._logger.info(f'TestCoding.test_wpc_embed_extract({q=}, {seed=})')
        rng = np.random.default_rng(seed)
        x = rng.integers(0, q, (32, 32), dtype='uint8')
        wet = rng.random(x.shape) < .5
        rho = np.where(wet, np.inf, 1.)
        message = rng.bytes(16)
        k = cl.coding.wpc.embed(x, (rho,)*(q-1), message, seed=seed)
        self.assertEqual(k.shape, x.shape)
        self.assertTrue((k < q).all())
        y = (x + k) % q
        self.assertEqual(cl.coding.wpc.extract(y, 16, seed=seed, q=q), message)
        # wet unchanged
        np.testing.assert_array_equal(k[wet], 0)

    @parameterized.expand([
        [2, 8, .25], [2, 18, .1], [3, 6, .25], [3, 11, .1], [5, 5, .1],
    ])
    def test_wpc_efficiency(self, q, p, alpha):
        self._logger.info(f'TestCoding.test_wpc_efficiency({q=}, {p=}, {alpha=})')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, q, 8192)
        wet = rng.random(x.shape) < .5
        rho = np.where(wet, np.inf, 1.)
        length = int(alpha * (~wet).sum()) // 8
        message = rng.bytes(length)
        k = cl.coding.wpc.embed(x, (rho,)*(q-1), message, block_size=p)
        y = (x + k) % q
        self.assertEqual(cl.coding.wpc.extract(y, length, q=q, block_size=p), message)
        np.testing.assert_array_equal(k[wet], 0)
        # fewest changes, better than 2 bits per change of an arbitrary solution
        self.assertGreater(8 * length / (k != 0).sum(), 2.5)

    @parameterized.expand([[q, block_size] for q in [2, 3] for block_size in [10, None]])
    def test_wpc_blocks(self, q, block_size):
        self._logger.info(f'TestCoding.test_wpc_blocks({q=}, {block_size=})')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, q, (64, 64))
        # wet elements clustered in top half
        wet = np.zeros(x.shape, dtype=bool)
        wet[:32] = True
        wet[32:] = rng.random((32, 64)) < .3
        rho = np.where(wet, np.inf, 1.)
        message = rng.bytes(64)  # 512 bits or 324 trits
        # random order spreads wet elements over blocks
        order = np.random.default_rng(1).permutation(x.size)
        x, rho, wet = x.flatten()[order], rho.flatten()[order], wet.flatten()[order]
        k = cl.coding.wpc.embed(x, (rho,)*(q-1), message, seed=12345, block_size=block_size)
        y = (x + k) % q
        self.assertEqual(cl.coding.wpc.extract(y, 64, seed=12345, q=q, block_size=block_size), message)
        np.testing.assert_array_equal(k[wet], 0)

    @parameterized.expand([[2, None], [2, 10], [3, None], [3, 7], [5, None]])
    def test_wpc_backends(self, q, block_size):
        self._logger.info(f'TestCoding.test_wpc_backends({q=}, {block_size=})')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, q, 4096, dtype='uint8')
        # about half of the elements wet
        wet = 1 - .5**(1 / (q-1))
        rhos = [np.where(rng.random(x.shape) < wet, np.inf, rng.random(x.shape)) for _ in range(q-1)]
        message = rng.bytes(64)
        # same changes and message with both backends
        with cl.tools.BACKEND_PYTHON:
            k_py = cl.coding.wpc.embed(x, rhos, message, seed=1, block_size=block_size)
        with cl.tools.BACKEND_RUST:
            k_rs = cl.coding.wpc.embed(x, rhos, message, seed=1, block_size=block_size)
        np.testing.assert_array_equal(k_rs, k_py)
        y = (x + k_rs) % q
        for backend in [cl.tools.BACKEND_PYTHON, cl.tools.BACKEND_RUST]:
            with backend:
                self.assertEqual(cl.coding.wpc.extract(y, 64, seed=1, q=q, block_size=block_size), message)

    @parameterized.expand([[cl.tools.BACKEND_PYTHON], [cl.tools.BACKEND_RUST]])
    def test_wpc_backends_too_wet(self, backend):
        self._logger.info(f'TestCoding.test_wpc_backends_too_wet({backend=})')
        x = np.zeros(4096, dtype='uint8')
        rho = np.where(np.arange(4096) < 2048, np.inf, 1.)
        with backend, self.assertRaisesRegex(ValueError, 'block 0: 16 symbols do not fit into 0 dry elements'):
            cl.coding.wpc.embed(x, rho, b'\xff' * 8, block_size=16)

    def test_wpc_blocks_clustered(self):
        self._logger.info('TestCoding.test_wpc_blocks_clustered()')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 2, 4096)
        # wet elements clustered in first half, blocks there are too wet
        rho = np.where(np.arange(4096) < 2048, np.inf, 1.)
        with self.assertRaises(ValueError):
            cl.coding.wpc.embed(x, rho, rng.bytes(64), seed=12345, block_size=16)

    @parameterized.expand([[2, 23], [3, 14]])
    def test_wpc_block_size(self, q, block_size):
        self._logger.info(f'TestCoding.test_wpc_block_size({q=}, {block_size=})')
        x = np.zeros(64, dtype='uint8')
        with self.assertRaises(ValueError):  # q^block_size too large
            cl.coding.wpc.embed(x, (np.ones(64),)*(q-1), b'a', block_size=block_size)
        with self.assertRaises(ValueError):
            cl.coding.wpc.extract(x, 1, q=q, block_size=block_size)

    def test_wpc_wet_cost(self):
        self._logger.info('TestCoding.test_wpc_wet_cost()')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 2, (32, 32))
        wet = rng.random(x.shape) < .5
        rho = np.where(wet, 10**13, 1.)  # ebs-like wet cost
        message = rng.bytes(16)
        k = cl.coding.wpc.embed(x, rho, message, seed=12345)
        self.assertEqual(cl.coding.wpc.extract(x ^ k, 16, seed=12345), message)
        np.testing.assert_array_equal(k[wet], 0)

    def test_wpc_ternary_half_wet(self):
        self._logger.info('TestCoding.test_wpc_ternary_half_wet()')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 3, (32, 32))
        # element wet in one direction only is wet
        rho_p1 = np.where(rng.random(x.shape) < .2, np.inf, 1.)
        rho_m1 = np.where(rng.random(x.shape) < .2, np.inf, 1.)
        message = rng.bytes(16)
        k = cl.coding.wpc.embed(x, (rho_p1, rho_m1), message, seed=12345)
        self.assertEqual(cl.coding.wpc.extract((x + k) % 3, 16, seed=12345, q=3), message)
        np.testing.assert_array_equal(k[np.isinf(rho_p1) | np.isinf(rho_m1)], 0)

    @parameterized.expand([[2], [3]])
    def test_wpc_add_zero(self, q):
        self._logger.info(f'TestCoding.test_wpc_add_zero({q=})')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, q, (32, 32))
        rhos = [np.where(rng.random(x.shape) < .3, np.inf, rng.random(x.shape)) for _ in range(q-1)]
        message = rng.bytes(16)
        # explicit zero-cost no change is the same as implicit
        k = cl.coding.wpc.embed(x, rhos, message, seed=12345)
        k_explicit = cl.coding.wpc.embed(x, [np.zeros(x.shape)] + rhos, message, seed=12345, add_zero=False)
        np.testing.assert_array_equal(k_explicit, k)
        self.assertEqual(cl.coding.wpc.extract((x + k) % q, 16, seed=12345, q=q), message)
        # no change cannot be wet
        with self.assertRaises(ValueError):
            cl.coding.wpc.embed(x, [np.full(x.shape, np.inf)] + rhos, message, seed=12345, add_zero=False)

    def test_wpc_public(self):
        self._logger.info('TestCoding.test_wpc_public()')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 2, 4096)
        rho = np.where(rng.random(x.shape) < .5, np.inf, 1.)
        message = rng.bytes(64)
        # default public matrix, shared by all blocks
        k = cl.coding.wpc.embed(x, rho, message)
        self.assertEqual(cl.coding.wpc.extract(x ^ k, 64), message)

    def test_wpc_wrong_key(self):
        self._logger.info('TestCoding.test_wpc_wrong_key()')
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 2, (32, 32))
        rho = np.ones(x.shape)
        message = rng.bytes(16)
        y = x ^ cl.coding.wpc.embed(x, rho, message, seed=12345, block_size=8)
        self.assertNotEqual(cl.coding.wpc.extract(y, 16, seed=54321, block_size=8), message)
        self.assertNotEqual(cl.coding.wpc.extract(y, 16, seed=12345, block_size=16), message)

    def test_wpc_too_wet(self):
        self._logger.info('TestCoding.test_wpc_too_wet()')
        x = np.zeros((16, 16), dtype='uint8')
        rho = np.ones(x.shape)
        rho[:15] = np.inf  # 16 dry elements
        with self.assertRaises(ValueError):
            cl.coding.wpc.embed(x, rho, b'abc', seed=12345)  # 24 bits

    def test_wpc_not_symbols(self):
        self._logger.info('TestCoding.test_wpc_not_symbols()')
        x = np.arange(16)
        with self.assertRaises(ValueError):
            cl.coding.wpc.embed(x, np.ones(16), b'a', seed=12345)
        with self.assertRaises(ValueError):
            cl.coding.wpc.extract(x, 1, seed=12345)

    def test_wpc_empty(self):
        self._logger.info('TestCoding.test_wpc_empty()')
        x = np.zeros(16, dtype='uint8')
        k = cl.coding.wpc.embed(x, np.ones(16), b'', seed=12345)
        np.testing.assert_array_equal(k, 0)
        self.assertEqual(cl.coding.wpc.extract(x, 0, seed=12345), b'')

    @parameterized.expand([[alpha] for alpha in [1e-3, .01, .05, .1, .2, .4]])
    def test_hamming_efficiency(self, alpha):
        self._logger.info('TestCoding.test_hamming_efficiency')
        e_hamming = cl.coding.hamming.efficiency(alpha)
        e_bound = cl.coding.efficiency(alpha)
        self.assertGreater(e_bound, e_hamming)


__all__ = ["TestCoding"]
