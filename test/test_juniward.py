
import conseal as cl
import jpeglib
import logging
from parameterized import parameterized
import unittest
import numpy as np
from scipy.io import loadmat

from .defs import ASSETS_DIR
STEGO_DIR = ASSETS_DIR / 'juniward'
COVER_DIR = ASSETS_DIR / 'cover'


class TestJUNIWARD(unittest.TestCase):
    '''Test suite for J-UNIWARD embedding.'''
    _logger = logging.getLogger(__name__)

    @staticmethod
    def read_costmap(costmap_filename, height, width):
        '''
        Read binary file produced by a modified J-UNIWARD C++ implementation
        :param costmap_filename: path to binary file
        :param height: of the original image
        :param width: of the original image
        :return: ndarray of shape [height, width, 3]
            Channel 0: If the cover pixel is at the minimum -1023 already, the pixel contains the wet cost; otherwise rho.
            Channel 1: Always 0.
            Channel 2: If the cover pixel is at the maximum 1023 already, the pixel contains the wet cost; otherwise rho.
        '''
        count = height * width * 3
        with open(costmap_filename, 'rb') as f:
            costmap = np.fromfile(f, dtype=np.float32, count=count, sep='')
            costmap = costmap.reshape(height, width, 3)
        return costmap

    @parameterized.expand([
        # ('lizard_gray.jpeg', 'costmap-cpp-original/lizard_gray.costmap', cl.juniward.Implementation.JUNIWARD_ORIGINAL),
        # ('mountain_gray.jpeg', 'costmap-cpp-original/mountain_gray.costmap', cl.juniward.Implementation.JUNIWARD_ORIGINAL),
        # ('nuclear_gray.jpeg', 'costmap-cpp-original/nuclear_gray.costmap', cl.juniward.Implementation.JUNIWARD_ORIGINAL),
        ('lizard_gray.jpeg', 'lizard_gray_fix.costmap', cl.juniward.Implementation.JUNIWARD_FIX_OFF_BY_ONE),
        ('mountain_gray.jpeg', 'mountain_gray_fix.costmap', cl.juniward.Implementation.JUNIWARD_FIX_OFF_BY_ONE),
        ('nuclear_gray.jpeg', 'nuclear_gray_fix.costmap', cl.juniward.Implementation.JUNIWARD_FIX_OFF_BY_ONE),
    ])
    def test_costmap_cpp_python_equivalence(self, cover_filename, costmap_cpp_filename, implementation):
        self._logger.info(f'TestJUNIWARD.test_costmap_fix_cpp_python_equivalence({cover_filename=}, {costmap_cpp_filename=}, {implementation=})')

        cover_filepath = COVER_DIR / cover_filename
        costmap_cpp_filepath = STEGO_DIR / 'costmap-cpp' / costmap_cpp_filename

        cover_spatial = np.squeeze(jpeglib.read_spatial(cover_filepath).spatial[..., 0]).astype(np.float64)
        img_dct = jpeglib.read_dct(cover_filepath)
        cover_dct_coeffs = img_dct.Y
        quantization_table = img_dct.qt[0]

        # Compute cost for embedding into the quantized DCT coefficients of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
        wet_cost = 10 ** 13
        rho_p1, rho_m1 = cl.juniward.compute_cost_adjusted(
            cover_spatial=cover_spatial,
            cover_dct_coeffs=cover_dct_coeffs,
            quantization_table=quantization_table,
            dtype=np.float64,
            implementation=implementation,
            wet_cost=wet_cost,
        )

        # Rearrange from 4D to 2D
        rho_p1_2d = cl.tools.dct.jpeglib_to_jpegio(rho_p1)
        rho_m1_2d = cl.tools.dct.jpeglib_to_jpegio(rho_m1)

        costmap_cpp = self.read_costmap(costmap_cpp_filepath, 512, 512)

        # Set rtol and atol to the default values used by np.allclose()
        np.testing.assert_allclose(rho_m1_2d, costmap_cpp[:, :, 0], rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(rho_p1_2d, costmap_cpp[:, :, 2], rtol=1e-05, atol=1e-08)

    @parameterized.expand([
        ('lizard_gray.jpeg', 'lizard_gray_costmap_original.mat', cl.JUNIWARD_ORIGINAL),
        ('mountain_gray.jpeg', 'mountain_gray_costmap_original.mat', cl.JUNIWARD_ORIGINAL),
        ('nuclear_gray.jpeg', 'nuclear_gray_costmap_original.mat', cl.JUNIWARD_ORIGINAL),
        ('lizard_gray.jpeg', 'lizard_gray_costmap_fix.mat', cl.JUNIWARD_FIX_OFF_BY_ONE),
        ('mountain_gray.jpeg', 'mountain_gray_costmap_fix.mat', cl.JUNIWARD_FIX_OFF_BY_ONE),
        ('nuclear_gray.jpeg', 'nuclear_gray_costmap_fix.mat', cl.JUNIWARD_FIX_OFF_BY_ONE),
    ])
    def test_costmap_matlab_python_equivalence(self, cover_filename, costmap_matlab_filename, implementation):
        self._logger.info(f'TestJUNIWARD.test_costmap_matlab_python_equivalence({cover_filename=}, {costmap_matlab_filename=}, {implementation=})')

        cover_filepath = COVER_DIR / cover_filename
        costmap_matlab_filepath = STEGO_DIR / 'costmap-matlab' / costmap_matlab_filename

        img_spatial = np.squeeze(jpeglib.read_spatial(cover_filepath).spatial[..., 0]).astype(np.float64)
        img_dct = jpeglib.read_dct(cover_filepath)
        qt = img_dct.qt[0]

        costmap = cl.juniward._costmap.compute_cost(
            spatial=img_spatial,
            quantization_table=qt,
            implementation=implementation,
        )

        # Convert from 4D to 2D
        costmap = cl.tools.dct.jpeglib_to_jpegio(costmap)

        # Compare to reference
        costmap_matlab = loadmat(costmap_matlab_filepath)['rho']
        np.testing.assert_allclose(costmap, costmap_matlab)

    @parameterized.expand([
        ('lizard_gray.jpeg', 'lizard_gray_matlab_alpha_0.4_seed_6020.jpeg', 0.4, 6020),
        ('mountain_gray.jpeg', 'mountain_gray_matlab_alpha_0.4_seed_6020.jpeg', 0.4, 6020),
        ('nuclear_gray.jpeg', 'nuclear_gray_matlab_alpha_0.4_seed_6020.jpeg', 0.4, 6020),
    ])
    def test_simulation_python_matlab_equivalence(self, cover_filename, stego_filename, embedding_rate, seed):
        self._logger.info(f'TestJUNIWARD.test_simulation_python_matlab_equivalence('f'{cover_filename=}, {embedding_rate=}, {seed=})')

        cover_filepath = COVER_DIR / cover_filename
        stego_matlab_filepath = STEGO_DIR / 'stego-matlab' / stego_filename

        # Read grayscale image
        cover_spatial = jpeglib.read_spatial(cover_filepath).spatial[:, :, 0]
        cover_spatial = cover_spatial.astype(np.float64)

        # Read DCT coefficients and quantization table
        img_dct = jpeglib.read_dct(cover_filepath)
        cover_dct_coeffs = img_dct.Y
        qt = img_dct.qt[0]

        # Simulate stego embedding using fixed seed and generator
        stego_dct_coeffs = cl.juniward.simulate_single_channel(
            cover_spatial=cover_spatial,
            cover_dct_coeffs=cover_dct_coeffs,
            quantization_table=qt,
            embedding_rate=embedding_rate,
            implementation=cl.JUNIWARD_ORIGINAL,
            generator='MT19937',
            seed=seed)

        # Read stego images created using Matlab
        stego_matlab_im = jpeglib.read_dct(stego_matlab_filepath)
        stego_matlab_dct_coeffs = stego_matlab_im.Y

        # Compare stego images
        np.testing.assert_allclose(stego_dct_coeffs, stego_matlab_dct_coeffs)


__all__ = ['TestJUNIWARD']
