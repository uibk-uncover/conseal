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

The code before use a cover image, which can be loaded as follows.

>>> import jpeglib
>>> im_dct = jpeglib.read_dct("cover.jpeg")
>>> im_spatial = jpeglib.read_spatial(  # for J-UNIWARD
...   "cover.jpeg",
...   jpeglib.JCS_GRAYSCALE)

After modifying ``im_dct.Y``, write the image as follows.

>>> im_dct.write_dct("stego.jpeg")

.. note::

   ``conseal`` expects the DCT coefficients in 4D shape [num_vertical_blocks, num_horizontal_blocks, 8, 8].
   If you use to 2D DCT representation (as used by jpegio, for instance),
   you have to convert it to 4D and back as follows.

   >>> dct_coeffs_4d = (dct_coeffs_2d  # 4D to 2D
   ...   .reshape(dct_coeffs_2d.shape[0]//8, 8, dct_coeffs_2d.shape[1]//8, 8)
   ...   .transpose(0, 2, 1, 3))

   >>> dct_coeffs_2d = (dct_coeffs_4d  # 4D to 2D
   ...   .transpose(0, 2, 1, 3)
   ...   .reshape(dct_coeffs_4d.shape[0]*8, dct_coeffs_4d.shape[1]*8))


The ``conseal`` API provides methods on different levels of abstraction.
Lower level API calls provide more control, but its use requires more code.
In any case, please make yourself familiar with the codebase before using it!


At high-level API
-----------------

Using the high-level API, you can obtain the stego image from a cover image with a single function call.

>>> im_dct.Y = cl.nsF5.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,  # quantized cover DCT coefficients
...   quantization_table=im_dct.qt[0],  # QT
...   embedding_rate=0.4,  # alpha
...   seed=12345)  # seed

>>> im_dct.Y = cl.uerd.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0],  # QT
...   embedding_rate=0.4,  # alpha
...   seed=12345)  # seed

>>> im_dct.Y = cl.juniward.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0],  # QT
...   cover_spatial=im_spatial.spatial[..., 0],  # decompressed
...   embedding_rate=0.4,  # alpha
...   seed=12345)  # seed

J-UNIWARD requires the decompressed image for computing the distortion. Hence, the decompressed image must be provided alongside the DCT coefficients.


At mid-level API
----------------

Mid-level API exposes the separation principle.
It allows user to separately calculate the distortion, and perform the simulation of coding.

>>> rho_p1, rho_m1 = cl.juniward.compute_cost_adjusted(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0],  # QT
...   cover_spatial=im_spatial.spatial[..., 0])  # pixels
>>> im_dct.Y += cl.simulate.ternary(
...   rho_p1=rho_p1,  # distortion of +1
...   rho_m1=rho_m1,  # distortion of -1
...   alpha=0.4,  # alpha
...   n=im_dct.Y.size,  # cover size
...   seed=12345)  # seed

>>> rho_p1, rho_m1 = cl.uerd.compute_cost_adjusted(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0])  # QT
>>> im_dct.Y += cl.simulate.ternary(
...   rho_p1=rho_p1,  # distortion of +1
...   rho_m1=rho_m1,  # distortion of -1
...   alpha=0.4,  # alpha
...   n=im_dct.Y.size,  # cover size
...   seed=12345)  # seed

Notice that unlike the high-level API, the mid-level and low-level API return only the steganography noise, which is to be added to the cover.

At low-level API
----------------

The low-level API allows accessing the raw costs (without wet cost modification),
as well as the probabilities and simulation.

>>> rho = cl.uerd._costmap.compute_cost(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0])  # QT
>>> # ... (sanitize rho, create rho_p1 and rho_m1)
>>> (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
...   rho_p1=rho_p1,  # distortion of +1
...   rho_m1=rho_m1,  # distortion of -1
...   alpha=0.4,  # alpha
...   n=im_dct.Y.size)  # cover size
>>> im_dct.Y += cl.simulate._ternary.simulate(
...   p_p1=p_p1,  # probability of +1
...   p_m1=p_m1,  # probability of -1
...   seed=12345)  # seed(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0])  # QT
>>> # ... (sanitize rho, create rho_p1 and rho_m1)
>>> (p_p1, p_m1), lbda = cl.simulate._ternary.probability(
...   rho_p1=rho_p1,  # distortion of +1
...   rho_m1=rho_m1,  # distortion of -1
...   alpha=0.4,  # alpha
...   n=im_dct.Y.size)  # cover size
>>> im_dct.Y += cl.simulate._ternary.simulate(
...   p_p1=p_p1,  # probability of +1
...   p_m1=p_m1,  # probability of -1
...   seed=12345)  # seed

The low-level API allows receiving the lambda parameter, which can be used
to estimate the average payload embedded into the image
as well as the probabilities and simulation.

>>> alpha_hat = cl.simulate._ternary.average_payload(
...   lbda=lbda,  # lambda (optimized)
...   rho_p1=rho_p1,  # distortion of +1
...   rho_m1=rho_m1)  # distortion of -1
