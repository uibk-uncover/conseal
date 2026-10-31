
import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import tempfile
import unittest

from . import defs


class TestColor(unittest.TestCase):
    """Test suite for color module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_concatenate_binary(self, fname: str):
        """Test color concatenate, binary."""
        self._logger.info(f'TestColor.test_concatenate_binary({fname})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))
        # compute hill distortion
        rho_pm1 = cl.color.map_channels(cl.hill._costmap.compute_cost, x0, stack_axis=None)
        # simulate
        (p_pm1,), _ = cl.color.concatenate.probability(rhos=(rho_pm1,), alpha=.4)
        delta = cl.color.concatenate.sample((p_pm1,), seed=12345)
        self.assertEqual(delta.shape, x0.shape)
        # compute embedding rate
        alpha_hat = np.sum([cl.tools.entropy(p) for p in p_pm1]) / x0.size
        self.assertAlmostEqual(alpha_hat, .4, places=3)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_concatenate_ternary(self, fname: str):
        """Test color concatenate, ternary."""
        self._logger.info(f'TestColor.test_concatenate_ternary({fname})')
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))

        # compute hill distortion
        rhos = cl.color.map_channels(cl.hill.compute_cost_adjusted, x0, stack_axis=None)
        rhos = cl.color.transpose01(rhos)
        # simulate
        alpha = .4
        ps, lbda = cl.color.concatenate.probability(
            rhos=rhos,
            alpha=alpha,
        )
        delta = cl.color.concatenate.sample(
            ps,
            seed=12345,
        )
        self.assertEqual(delta.shape, x0.shape)
        # compute embedding rates
        ns = cl.color.map_channels(lambda x: x.size, x0)
        alphas_hat = [
            cl.tools.entropy(ps[0][c], ps[1][c]) / ns[c]
            for c in range(x0.shape[-1])
        ]
        alpha_hat = np.sum([
            cl.tools.entropy(ps[0][c], ps[1][c])
            for c in range(x0.shape[-1])
        ]) / np.sum(ns)
        rhos_hat = [
            np.sum(ps[0][c] * rhos[0][c] + ps[1][c] * rhos[1][c]) / ns[c]
            for c in range(x0.shape[-1])
        ]
        np.testing.assert_allclose(alpha_hat, alpha, atol=1e-3)
        # print('joint3:', alphas, rhos)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_concatenate_ternary_jpeg(self, fname: str):
        self._logger.info(f'TestColor.test_concatenate_ternary_jpeg({fname})')
        # load cover
        jpeg = jpeglib.read_dct(defs.COVER_COMPRESSED_COLOR_DIR / f'{fname}.jpg')
        # compute distortion
        coefs = [jpeg.Y, jpeg.Cb, jpeg.Cr]
        rhos = [
            cl.uerd.compute_cost_adjusted(coefs[ch], jpeg.qt[jpeg.quant_tbl_no[ch]])
            for ch in range(3)
        ]
        rhos = cl.color.transpose01(rhos)

        # simulate
        alpha = .4
        (p_p1, p_m1), lbda = cl.color.concatenate.probability(
            rhos=rhos,
            alpha=alpha,
        )
        # compute embedding rates
        alphas = [
            cl.tools.entropy(p_p1[c], p_m1[c]) / coefs[c].size
            for c in range(3)
        ]
        # total alpha
        alpha_hat = np.sum([
            alphas[c] * coefs[c].size
            for c in range(3)
        ]) / np.sum([coef.size for coef in coefs])
        self.assertAlmostEqual(alpha, alpha_hat, places=3)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_concatenate_same_as_stacked(self, fname: str):
        """One lambda over the channels is the search over the concatenated costs."""
        self._logger.info(f'TestColor.test_concatenate_same_as_stacked({fname})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))
        rhos = cl.color.transpose01(cl.color.map_channels(cl.hill.compute_cost_adjusted, x0, stack_axis=None))
        stacked = (np.stack(rhos[0]), np.stack(rhos[1]))
        # DDE: same as the stacked search, and as the former joint
        (p_p1, p_m1), lbda = cl.color.concatenate.probability(rhos=rhos, alpha=.4)
        (q_p1, q_m1), lbda_stacked = cl.simulator.probability(stacked, alpha=.4, n=x0.size)
        self.assertEqual(lbda, lbda_stacked)
        np.testing.assert_array_equal(np.stack(p_p1), q_p1)
        np.testing.assert_array_equal(np.stack(p_m1), q_m1)
        # other solvers, the payload is met
        (p_p1, p_m1), _ = cl.color.concatenate.probability(rhos=rhos, alpha=.4, solver=cl.simulator.SOLVER_NEWTON)
        alpha_hat = np.sum([cl.tools.entropy(p_p1[c], p_m1[c]) for c in range(3)]) / x0.size
        self.assertAlmostEqual(alpha_hat, .4, places=3)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_concatenate_jpeg_shapes(self, fname: str):
        """Channels of different shapes, e.g. subsampled chroma."""
        self._logger.info(f'TestColor.test_concatenate_jpeg_shapes({fname})')
        jpeg = jpeglib.read_dct(defs.COVER_COMPRESSED_COLOR_DIR / f'{fname}.jpg')
        coefs = [jpeg.Y, jpeg.Cb, jpeg.Cr]
        rhos = cl.color.transpose01([
            cl.uerd.compute_cost_adjusted(coefs[ch], jpeg.qt[jpeg.quant_tbl_no[ch]])
            for ch in range(3)
        ])
        (p_p1, p_m1), _ = cl.color.concatenate.probability(rhos=rhos, alpha=.4)
        for ch in range(3):
            self.assertEqual(p_p1[ch].shape, coefs[ch].shape)
        n = np.sum([c.size for c in coefs])
        alpha_hat = np.sum([cl.tools.entropy(p_p1[c], p_m1[c]) for c in range(3)]) / n
        self.assertAlmostEqual(alpha_hat, .4, places=3)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_concatenate_simulate_as_former(self, fname: str):
        """Sampling gives the changes of the former simulator, for the same seed."""
        self._logger.info(f'TestColor.test_concatenate_simulate_as_former({fname})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))
        rhos = cl.color.transpose01(cl.color.map_channels(cl.hill.compute_cost_adjusted, x0, stack_axis=None))
        ps, _ = cl.color.concatenate.probability(rhos=rhos, alpha=.4)
        delta = cl.color.concatenate.sample(ps, seed=12345)
        seeds = np.random.default_rng(12345).integers(low=0, high=2**31, dtype=np.int32, size=3)
        former = np.stack([
            cl.simulate_old._ternary.simulate((ps[0][ch], ps[1][ch]), seed=seeds[ch])
            for ch in range(3)
        ], axis=-1)
        np.testing.assert_array_equal(delta, former)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_independent_simulate(self, fname: str):
        """Test color joint."""
        self._logger.info("TestColor.test_independent_simulate")
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))
        #
        alpha = .4
        hs_hat, rhos_hat = [], []
        for ch in range(x0.shape[-1]):
            # cost
            (rho_p1, rho_m1) = cl.hill.compute_cost_adjusted(x0[..., ch])
            n_ch = x0[..., ch].size
            # simulate
            (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
                rhos=(rho_p1, rho_m1),
                alpha=alpha,
                n=n_ch,
            )
            delta = cl.simulate._ternary.simulate(
                ps=(p_p1, p_m1),
                seed=12345,
            )
            #
            hs_hat.append(
                cl.tools.entropy(p_p1, p_m1)
            )
        alpha_hat = np.sum(hs_hat) / x0.size
        self.assertAlmostEqual(alpha_hat, alpha, places=3)
        # print('independent:', alphas_hat, rhos)

    @parameterized.expand([[True], [False]])
    def test_joint_full_circle(self, symmetric: bool):
        """Costs and probabilities go from marginal to joint and back."""
        self._logger.info(f'TestColor.test_joint_full_circle({symmetric})')
        rng = np.random.default_rng(12345)
        k = 1 if symmetric else 2
        rhos = tuple(rng.random((16, 16, 3)) for _ in range(k))
        ps = tuple(rng.random((16, 16, 3)) / (k + 1) for _ in range(k))
        #
        rhos_joint = cl.color.joint.from_marginal(rhos)
        self.assertEqual(rhos_joint.shape, ((k + 1)**3 - 1, 16, 16))
        rhos_hat = cl.color.joint.to_marginal(rhos_joint, symmetric=symmetric)
        for rho, rho_hat in zip(rhos, rhos_hat, strict=True):
            np.testing.assert_allclose(rho_hat, rho)
        np.testing.assert_allclose(cl.color.joint.from_marginal(rhos_hat), rhos_joint)
        #
        ps_joint = cl.color.joint.from_marginal(ps=ps)
        ps_hat = cl.color.joint.to_marginal(ps=ps_joint, symmetric=symmetric)
        for p, p_hat in zip(ps, ps_hat, strict=True):
            np.testing.assert_allclose(p_hat, p)

    @parameterized.expand([
        [f, symmetric]
        for f in defs.TEST_IMAGES
        for symmetric in [True, False]
    ])
    def test_joint_embedding_rate(self, fname: str, symmetric: bool):
        """Joint embedding over expanded costs gives the marginal embedding rate."""
        self._logger.info(f'TestColor.test_joint_embedding_rate({fname}, {symmetric})')
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))[:128, :128]
        if symmetric:
            rhos = (cl.color.map_channels(cl.hill._costmap.compute_cost, x0),)
        else:
            rhos = cl.color.map_channels(cl.hill.compute_cost_adjusted, x0)  # [2, H, W, 3]
        # joint embedding
        alpha = .4
        rhos_joint = cl.color.joint.from_marginal(rhos)
        ps_joint, _ = cl.simulator.probability(rhos_joint, alpha=alpha, n=x0.size)
        # marginal embedding rate
        ps = cl.color.joint.to_marginal(ps=ps_joint, symmetric=symmetric)
        alpha_hat = np.sum([
            cl.tools.entropy(*[p[..., c] for p in ps])
            for c in range(x0.shape[-1])
        ]) / x0.size
        self.assertAlmostEqual(alpha_hat, alpha, places=3)


__all__ = ["TestColor"]
