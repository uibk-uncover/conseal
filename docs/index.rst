conseal
=======

**conseal** is a Python package, containing implementations of modern image steganographic algorithms.

.. note::

   This project is under active development.

Simulating steganography has never been easier!

>>> import jpeglib
>>> import conseal as cl
>>> im_dct = jpeglib.read_dct("cover.jpeg")
>>> im_px = jpeglib.read_spatial("cover.jpeg", jpeglib.JCS_GRAYSCALE)
>>> im_dct.Y = cl.juniward.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,
...   quantization_table=im_dct.qt[0],
...   cover_spatial=im_px.spatial[..., 0]
...   embedding_rate=0.4,
...   seed=12345)
>>> im_dct.write_dct("stego.jpeg")

.. list-table:: Available embedding simulators.
   :widths: 25 75
   :header-rows: 1

   * - Domain
     - Algorithms
   * - DCT
     - J-UNIWARD, UERD, nsF5

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   reference
   glossary
   faq