conseal
=======

**conseal** is a Python package, containing implementations of modern image steganographic algorithms.

.. note::

   This project is under active development.

Simulating steganography has never been easier!

>>> # load JPEG cover
>>> import jpeglib
>>> im0 = jpeglib.read_spatial("cover.jpeg")
>>> jpeg = jpeglib.read_dct("cover.jpeg")
>>>
>>> # simulate JPEG steganography
>>> import conseal as cl
>>> jpeg.Y = cl.juniward.simulate_single_channel(
...   y0=jpeg.Y,               # DCT of cover luminance
...   qt=jpeg.qt[0],           # quantization table
...   x0=im0.spatial[..., 0],  # decompressed pixels of cover luminance
...   alpha=0.4,               # embedding rate [bpc]
...   seed=12345)              # seed for PRNG
...
>>> jpeg.write_dct("stego.jpeg")


>>> # load spatial cover
>>> import numpy as np
>>> from PIL import Image
>>> x0 = np.array(Image.open("cover.png"))
>>>
>>> # simulate spatial steganography
>>> import conseal as cl
>>> x1 = cl.hill.simulate_single_channel(
...   x0=x0,       # cover pixels
...   alpha=0.4,   # embedding rate [bpp]
...   seed=12345)  # seed for PRNG
...
>>> # save stego
>>> Image.fromarray(x1).save("stego.png")


.. list-table:: Available embedding simulators.
   :widths: 25 75
   :width: 100%
   :header-rows: 1

   * - Domain
     - Algorithms
   * - DCT
     - EBS, F5, J-UNIWARD, LSB, nsF5, UERD
   * - Spatial
     - HILL, HUGO, LSB, MiPOD, S-UNIWARD, WOW

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   reference
   glossary
   faq