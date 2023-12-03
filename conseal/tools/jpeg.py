"""

Author: Martin Benes, Benedikt Lorch
Affiliation: University of Innsbruck
"""

import jpeglib
import numpy as np
import tempfile


def qf_to_qt(qf, libjpeg_version='6b'):
    """
    Obtain quantization table corresponding to a specific quality factor
    :param qf: JPEG quality factor
    :param libjpeg_version: version of libjpeg library, passed to jpeglib
    :return: list of quantization tables
    """
    dummy_img = np.random.randint(low=0, high=256, dtype=np.uint8, size=(64, 64, 3))

    im = jpeglib.from_spatial(dummy_img)

    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        with jpeglib.version(libjpeg_version):
            im.write_spatial(f.name, qt=qf)

        return get_qt_from_jpeglib(f.name)


def get_qt_from_jpeglib(filepath):
    """
    Loads quantization table from a JPEG image using jpeglib
    :param filepath: path to JPEG image
    :return: quantization table given by jpeglib
    """

    jpeg = jpeglib.read_dct(filepath)
    qt = jpeg.qt
    quant_tbl_no = jpeg.quant_tbl_no

    return qt
