conseal
=======

**conseal** is a Python package, containing implementations of modern image steganographic algorithms.

.. note::

   This project is under active development.

Simulating steganography has never been easier!

>>> import jpeglib; import conseal as cl
>>> im_dct = jpeglib.read_dct("cover.jpeg")
>>> img_spatial = jpeglib.read_spatial("cover.jpeg")
>>> im_dct.Y = cl.juniward.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0],  # QT
...   cover_spatial=im_spatial.spatial[..., 0],  # decompressed
...   embedding_rate=0.4,  # alpha
...   seed=12345)  # seed
>>> im_dct.write_dct("stego.jpeg")

.. list-table:: Available embedding simulators.
   :widths: 50 50
   :header-rows: 1

   * - Spatial
     - JPEG
   * -
     - J-UNIWARD
   * -
     - UERD
   * -
     - nsF5

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   reference
   glossary
   faq