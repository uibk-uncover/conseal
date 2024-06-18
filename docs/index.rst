conseal
=======

**conseal** is a Python package, containing implementations of modern image steganographic algorithms.

.. note::

   This project is under active development.

Simulating steganography has never been easier!

>>> # load JPEG cover
>>> import jpeglib
>>> im_dct = jpeglib.read_dct("cover.jpeg")
>>> im_px = jpeglib.read_spatial("cover.jpeg")
>>>
>>> # simulate JPEG steganography
>>> import conseal as cl
>>> im_dct.Y = cl.juniward.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,
...   quantization_table=im_dct.qt[0],
...   cover_spatial=im_px.spatial[..., 0],
...   embedding_rate=0.4,
...   seed=12345)
...
>>> # save stego
>>> im_dct.write_dct("stego.jpeg")


>>> # load spatial cover
>>> import numpy as np
>>> from PIL import Image
>>> cover_spatial = np.array(Image.open("cover.png"))
>>>
>>> # simulate spatial steganography
>>> import conseal as cl
>>> stego_spatial = cl.hill.simulate_single_channel(
...   cover_spatial=cover_spatial,
...   embedding_rate=0.4,
...   seed=12345)
...
>>> # save stego
>>> Image.fromarray(stego_spatial).save("stego.png")


.. list-table:: Available embedding simulators.
   :widths: 25 75
   :width: 100%
   :header-rows: 1

   * - Domain
     - Algorithms
   * - DCT
     - EBS, J-UNIWARD, LSB, nsF5, UERD
   * - Spatial
     - HILL, HUGO, LSB

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   reference
   glossary
   faq