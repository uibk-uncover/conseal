
import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from parameterized import parameterized
import scipy.io
import tempfile
import unittest

from . import defs

STEGO_DIR = defs.ASSETS_DIR / 'ebs'
COST_DIR = STEGO_DIR / 'costmap-matlab'


class TestEBS(unittest.TestCase):
    """Test suite for EBS embedding."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_cost_reference(self, f):
        self._logger.info(f'TestEBS.test_cost_reference({f})')
        # load cover
        jpeg = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{f}.jpg')

        # compute cost
        rho, _ = cl.ebs.compute_cost_adjusted(jpeg.Y, jpeg.qt[0], implementation=cl.EBS_ORIGINAL)

        # convert to compare
        rho = cl.tools.dct.jpeglib_to_jpegio(rho)

        # compare to Remi Cogranne's Matlab implementation
        rho_ref = scipy.io.loadmat(COST_DIR / f'{f}.mat', simplify_cells=True)['rhos']
        self.assertTrue(np.allclose(rho, rho_ref))

    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_payload(self, fname):
        self._logger.info(f'TestEBS.test_payload({fname})')
        # load cover
        jpeg = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')

        # probabilities, as in the simulator
        rhos = cl.ebs.compute_cost_adjusted(jpeg.Y, jpeg.qt[0], implementation=cl.EBS_ORIGINAL)
        nzAC = cl.tools.dct.nzAC(jpeg.Y)
        ps, _ = cl.simulator.probability(rhos, alpha=.4, n=nzAC)

        # payload in bits per nzAC, as round(payload*nzAC) in the Matlab reference
        np.testing.assert_allclose(cl.tools.entropy(*ps), .4 * nzAC, rtol=1e-2)

    @parameterized.expand([[f, i] for i, f in enumerate(defs.TEST_IMAGES, 1)])
    def test_stego_reference(self, fname, seed):
        """Compares to stego of the Matlab SI-EBS by Remi Cogranne, rewritten to plain EBS.

        The reference code is licensed and not included,
        get SI_EBS.m from https://codeocean.com/capsule/8552917.
        The stego objects seal{i}_alpha_0.4_seed_{i}.jpg were produced
        from the covers in jpeg_75_gray with payload 0.4 and seed i, after rewriting

        - DCT_rounded taken from the cover, instead of the DCT of the precover,
          i.e., no rounding error e_ri and no random sign of change
        - rho_f = qi.^2, without the term (0.5 - abs(e_ri))
        - DC and modes (1,5), (5,1), (5,5) of each block wet, as in EBS.m of Remi Cogranne,
          which is the reference of test_compute_cost
        - rhoP1 wet for DCT_rounded > 1022, rhoM1 wet for DCT_rounded < -1022
        - binary LSB simulator replaced by the ternary simulator from uerd.m (see test/assets/uerd),
          i.e., the same calc_lambda over ternary entropy, with
          RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', seed)),
          randChange = transpose(rand(flip(size(x)))), +1 for randChange < pChangeP1,
          -1 for pChangeP1 <= randChange < pChangeP1 + pChangeM1
        - payload round(payload * nzAC) kept, the same as conseal's bits per nzAC

        This matches conseal with generator='MT19937', which reproduces Matlab's mt19937ar.
        """
        self._logger.info(f'TestEBS.test_stego_reference({fname}, {seed})')
        # load cover
        jpeg = jpeglib.read_dct(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')

        # simulate, with Matlab generator
        y1 = cl.ebs.simulate_single_channel(
            jpeg.Y,
            jpeg.qt[0],
            .4,
            implementation=cl.EBS_ORIGINAL,
            generator='MT19937',
            seed=seed,
        )

        # compare to plain EBS rewritten from SI-EBS by Remi Cogranne, see above
        y1_ref = jpeglib.read_dct(STEGO_DIR / f'{fname}_alpha_0.4_seed_{seed}.jpg').Y
        np.testing.assert_array_equal(y1, y1_ref)


__all__ = ["TestEBS"]
