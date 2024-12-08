Usage
=====

.. contents:: Table of Contents
   :local:
   :depth: 1

Installation and setup
----------------------

To use ``conseal``, first install it using pip:

.. code-block:: console

   $ pip3 install conseal

Import the package with

>>> import conseal as cl

For steganography on pixels, load the cover image using `pillow` and `numpy`.

>>> import numpy as np
>>> from PIL import Image
>>> x0 = np.array(Image.open("cover.png"))

After modification, save the stego image

>>> Image.fromarray(x1).save("stego.png")

For JPEG steganography, load the DCT coefficients using our sister project `jpeglib`.

>>> import jpeglib
>>> jpeg0 = jpeglib.read_dct("cover.jpeg")
>>> im0 = jpeglib.read_spatial(  # decompressed
...   "cover.jpeg",
...   jpeglib.JCS_GRAYSCALE)

After copying ``jpeg0`` to ``jpeg1`` and modifying ``jpeg``
After modifying ``jpeg0.Y``, write the image as follows.

>>> jpeg1.write_dct("stego.jpeg")


.. note::

   ``conseal`` expects the DCT coefficients in 4D shape [num_vertical_blocks, num_horizontal_blocks, 8, 8].
   If you use to 2D DCT representation (as used by jpegio, for instance),
   you have to convert it to 4D and back as follows.

   >>> y_4d = cl.tools.jpegio_to_jpeglib(y_2d)
   >>> y_2d = cl.tools.jpeglib_to_jpegio(y_4d)


The ``conseal`` API provides methods on different levels of abstraction.
Lower level API calls provide more control, but its use requires more code.
In any case, please make yourself familiar with the codebase before using it!


At high-level API
-----------------

Using the high-level API, you can obtain the stego image from a cover image with a single function call.

>>> jpeg1.Y = cl.uerd.simulate_single_channel(
...   y0=jpeg0.Y,  # quantized cover DCT
...   qt=jpeg0.qt[0],  # quantization table
...   alpha=0.4,  # embedding rate
...   seed=12345)  # seed for PRNG


At mid-level API
----------------

Mid-level API exposes the separation principle.
It allows user to separately calculate the distortion, and perform the simulation or coding.

>>> rho_p1, rho_m1 = cl.uerd.compute_cost_adjusted(
...   y0=jpeg0.Y,  # DCT
...   qt=jpeg0.qt[0])  # QT
>>> jpeg1.Y += cl.simulate.ternary(
...   rhos=(rho_p1, rho_m1),  # costs of +1 and -1 changes
...   alpha=0.4,  # embedding rate
...   n=jpeg0.Y.size,  # cover size
...   seed=12345)  # seed for PRNG


Notice that unlike the high-level API, the mid-level and low-level API return only the steganography noise, which is to be added to the cover.

At low-level API
----------------

The low-level API allows accessing the raw costs (without wet cost modification),
as well as the probabilities and simulation.

>>> rho = cl.uerd._costmap.compute_cost(
...   y0=jpeg0.Y,  # DCT
...   qt=jpeg0.qt[0])  # QT
>>> # ... (sanitize rho, create rho_p1 and rho_m1)
>>> (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
...   rhos=(rho_p1, rho_m1),  # embedding costs
...   alpha=0.4,  # embedding rate
...   n=jpeg0.Y.size)  # cover size
>>> jpeg1.Y += cl.simulate._ternary.simulate(
...   ps=(p_p1, p_m1),
...   seed=12345)  # seed for PRNG

The low-level API gives access to the ``lbda`` parameter, which is used
to estimate the average payload embedded into the image
as well as the probabilities and simulation.

>>> alpha_hat = cl.simulate._ternary.average_payload(
...   lbda=lbda,  # lambda (optimized)
...   rhos=(rho_p1, rho_m1))  # cost of +1 and -1 changes

Some embedding methods such as nsF5 and LSB have a low-level interface to get probabilities directly

>>> (p_p1, p_m1), _ = cl.nsF5._costmap.probability(
...   y0=im_dct.Y,  # DCT
...   alpha=0.4)  # alpha
>>> im_dct.Y += cl.simulate._ternary.simulate(
...   ps=(p_p1, p_m1),  # probability of change
...   seed=12345)  # seed for PRNG

>>> (p_p1, p_m1), _ = cl.lsb._costmap.probability(
...   x0,  # pixels
...   alpha=0.4)  # embedding rate
>>> stego_spatial = cover_spatial + cl.simulate._ternary.simulate(
...   ps=(p_p1, p_m1),  # probability of change
...   seed=12345)  # seed for PRNG
