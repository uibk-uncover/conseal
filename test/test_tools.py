
import conseal as cl
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import pandas as pd
import sys
import tempfile
import unittest

sys.path.append('test')
import defs


class TestTools(unittest.TestCase):
    """Test suite for tools module."""
    _logger = logging.getLogger(__name__)

    def test_password(self):
        self._logger.info('TestTools.test_password')
        # setup
        password1 = 'password123'
        password2 = 'password124'
        # convert to key
        key1a = cl.tools.password_to_seed(password1)
        key2 = cl.tools.password_to_seed(password2)
        key1b = cl.tools.password_to_seed(password1)
        self.assertEqual(key1a, key1b)  # same password = same key
        self.assertNotEqual(key1a, key2)  # different password = different key

    def test_permute_blocks(self):
        self._logger.info('TestTools.test_permute')
        # setup
        key = 123
        rng = np.random.default_rng(12345)
        dct = rng.integers(-1028, 1028, (4, 4, 8, 8), dtype='int16')
        visited = np.zeros(dct.shape, dtype='int')
        # random blocks
        blocks = cl.tools.blocks(dct, key=key)
        visited[blocks] += 1
        # all blocks visited
        self.assertTrue((visited == 1).all())

    def test_iterate_ac(self):
        self._logger.info('TestTools.test_iterate_ac')
        # setup
        key = 123
        rng = np.random.default_rng(12345)
        dct = rng.integers(-1028, 1028, (4, 4, 8, 8), dtype='int16')
        visited = np.zeros(dct.shape, dtype='int')
        # iterate
        for h, w, hb, wb in cl.tools.iterate_ac(dct, key=key):
            visited[h, w, hb, wb] += 1
        # all ACs visited
        acs = np.ones(visited.shape, dtype='bool')
        acs[:, :, 0, 0] = False
        self.assertTrue((visited[acs] == 1).all())
        # DCs not visited
        self.assertTrue((visited[~acs] == 0).all())

    def test_iterate_nzac(self):
        self._logger.info('TestTools.test_iterate_nzac')
        # setup
        key = 123
        rng = np.random.default_rng(12345)
        dct = rng.integers(-10, 10, (4, 4, 8, 8), dtype='int16')
        visited = np.zeros(dct.shape, dtype='int')
        # iterate
        for h, w, hb, wb in cl.tools.iterate_ac(dct, key=key):
            visited[h, w, hb, wb] += 1
        # all nzACs visited
        nzacs = np.ones(visited.shape, dtype='bool')
        nzacs[:, :, 0, 0] = False
        nzacs[dct == 0] = False
        self.assertTrue((visited[nzacs] == 1).all())
        # zero ACs and DCs not visited
        self.assertTrue((visited[~nzacs] == 0).all())

    @parameterized.expand([[None], [123]])
    def test_iterate_px(self, key: int):
        self._logger.info('TestTools.test_iterate_px')
        # setup
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 256, (3, 3, 3), dtype='uint8')
        visited = np.zeros(x.shape, dtype='int')
        # iterate
        for h, w, c in cl.tools.iterate(x, key=key):
            visited[h, w, c] += 1
        self.assertTrue((visited == 1).all())

    # @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_lrt(self):
        self._logger.info('TestTools.test_lrt')

        defls = []
        for f in defs.TEST_IMAGES:
            # load cover
            x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{f}.png'))
            defl = {}
            # LSB
            ps_lsb, _ = cl.lsb._costmap.probability(x0, alpha=.4, e=None)
            # print('LSB:', cl.simulate.average_payload(ps=ps_lsb)[1] / x0.size)
            defl['lsb'] = cl.tools.lrt.attack(x0, ps=ps_lsb, attacker=cl.ATTACKER_OMNISCIENT, clip=1e-4)
            # HUGO
            rhos_hugo = cl.hugo.compute_cost_adjusted(x0)
            ps_hugo, _ = cl.simulate._ternary.probability(rhos=rhos_hugo, alpha=.4, n=x0.size)
            # print('HUGO:', cl.simulate.average_payload(ps=ps_hugo)[1] / x0.size)
            defl['hugo'] = cl.tools.lrt.attack(x0, ps=ps_hugo, attacker=cl.ATTACKER_OMNISCIENT, clip=1e-4)
            # HILL
            rhos_hill = cl.hill.compute_cost_adjusted(x0)
            ps_hill, _ = cl.simulate._ternary.probability(rhos=rhos_hill, alpha=.4, n=x0.size)
            # print('HILL:', cl.simulate.average_payload(ps=ps_hill)[1] / x0.size)
            defl['hill'] = cl.tools.lrt.attack(x0, ps=ps_hill, attacker=cl.ATTACKER_OMNISCIENT, clip=1e-4)
            # MiPOD
            ps_mipod, _ = cl.mipod.probability(x0, alpha=.4)
            # print('MiPOD:', cl.simulate.average_payload(ps=ps_mipod)[1] / x0.size)
            defl['mipod'] = cl.tools.lrt.attack(x0, ps=ps_mipod, attacker=cl.ATTACKER_OMNISCIENT, clip=1e-4)
            #
            defls.append(defl)

        defls = pd.DataFrame(defls)
        print(defls)
        print(defls.median())


__all__ = ['TestTools']
