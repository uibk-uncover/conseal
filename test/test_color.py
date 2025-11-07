
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
    def test_joint_binary(self, fname: str):
        self._logger.info(f'TestColor.test_joint_binary({fname})')
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
        (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
            rhos=rhos,
            alpha=alpha,
            n=np.sum([coef.size for coef in coefs]),
            objective=cl.color.joint._ternary.average_payload,
        )
        # compute embedding rates
        alphas = [
            cl.tools.entropy(p_p1[c], p_m1[c]) / coefs[c].size
            for c in range(3)
        ]
        # total alpha
        alpha = np.sum([
            alphas[c] * coefs[c].size
            for c in range(3)
        ]) / np.sum([coef.size for coef in coefs])
        print(alpha)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_joint_simulate(self, fname: str):
        """Test color joint."""
        self._logger.info("TestColor.test_joint_simulate")
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))

        # compute hill distortion
        rhos = cl.color.map_channels(cl.hill.compute_cost_adjusted, x0, stack_axis=None)
        rhos = cl.color.transpose01(rhos)
        # simulate
        ps, lbda = cl.color.joint._ternary.probability(
            rhos=rhos,
            alpha=.4,
            n=x0.size,
        )
        delta = cl.color.joint._ternary.simulate(
            ps=ps,
            seed=12345,
        )
        # compute embedding rates
        ns = cl.color.map_channels(lambda x: x.size, x0)
        alphas = [
            cl.tools.entropy(ps[0][c], ps[1][c]) / ns[c]
            for c in range(x0.shape[-1])
        ]
        rhos = [
            np.sum(ps[0][c] * rhos[0][c] + ps[1][c] * rhos[1][c]) / ns[c]
            for c in range(x0.shape[-1])
        ]
        print('joint:', alphas, rhos)

    # @parameterized.expand([['mountain'], ['lizard'], ['nuclear']])
    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_independent_simulate(self, fname: str):
        """Test color joint."""
        self._logger.info("TestColor.test_independent_simulate")
        # load cover
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))
        #
        alphas, rhos = [], []
        for ch in range(x0.shape[-1]):
            # cost
            (rho_p1, rho_m1) = cl.hill.compute_cost_adjusted(x0[..., ch])
            n_ch = x0[..., ch].size
            # simulate
            (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
                rhos=(rho_p1, rho_m1),
                alpha=.4,
                n=n_ch,
            )
            delta = cl.simulate._ternary.simulate(
                ps=(p_p1, p_m1),
                seed=12345,
            )
            #
            alphas.append(
                cl.tools.entropy(p_p1, p_m1) / n_ch
            )
            rhos.append(
                np.sum(p_p1*rho_p1 + p_m1*rho_m1) / n_ch
            )

        print('independent:', alphas, rhos)

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_octary(self, fname: str):
        self._logger.info("TestColor.test_octary")
        # load cover
        # x0 = np.array(Image.open(f'img/{fname}.png'))
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'))

        # compute hill distortion
        rhos = cl.color.map_channels(cl.hill._costmap.compute_cost, x0)
        rhos[np.isinf(rhos) | np.isnan(rhos) | (rhos > 10**10)] = 10**10
        rhos = rhos.transpose(2, 0, 1)

        #
        rhos8, deltas = cl.color.q_to_pow2q(rhos)

        #
        delta = cl.color.qary(
            rhos=rhos8,
            deltas=deltas,
            alpha=.4,
            n=x0.size,
            seed=12345,
        )


__all__ = ["TestColor"]
