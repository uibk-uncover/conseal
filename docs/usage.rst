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

   ``conseal`` expects DCT tensor in 4D shape [block height, block width, 8, 8].
   If you use to 2D DCT representation (used by jpegio, for instance),
   you have to convert it to 4D and back as follows.

   >>> dct4 = (dct2  # 2D to 4D
   ...   .reshape(dct2.shape[0]//8, 8, dct.shape[1]//8, 8)
   ...   .transpose(0, 2, 1, 3))

   >>> dct2 = (dct4  # 4D to 2D
   ...   .transpose(0, 2, 1, 3)
   ...   .reshape(dct4.shape[0]*8, dct4.shape[1]*8))


Package ``conseal`` provides several API on different levels of abstraction.
The lower the level, the more control, but more verbose code becomes.


At high-level API
-----------------

Using the high-level API, you call a single function, provide the cover, and receive stego.

>>> im_dct.Y = cl.nsF5.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,  # DCT
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

J-UNIWARD computes the distortion in decompressed domain,
so it must be provided alongside the DCT coefficients.


At mid-level API
----------------

Mid-level API exposes the separation principle.
It allows user to separately calculate the distortion, and perform the simulation of coding.

>>> rho_p1, rho_m1 = cl.juniward.compute_distortion(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0],  # QT
...   cover_spatial=im_spatial.spatial[..., 0])  # pixels
>>> im_dct.Y += cl.simulate.ternary(
...   rho_p1=rho_p1,  # distortion of +1
...   rho_m1=rho_m1,  # distortion of -1
...   alpha=0.4,  # alpha
...   n=im_dct.Y.size,  # cover size
...   seed=12345)  # seed

>>> rho_p1, rho_m1 = cl.uerd.compute_distortion(
...   cover_dct_coeffs=im_dct.Y,  # DCT
...   quantization_table=im_dct.qt[0])  # QT
>>> im_dct.Y += cl.simulate.ternary(
...   rho_p1=rho_p1,  # distortion of +1
...   rho_m1=rho_m1,  # distortion of -1
...   alpha=0.4,  # alpha
...   n=im_dct.Y.size,  # cover size
...   seed=12345)  # seed

Notice that unlike high-level API, mid-level and low-level API return difference tensor,
that is to be added to the cover.


At low-level API
----------------

Low-level API allows accessing the raw costs (without wet cost modification),
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
...   seed=12345)  # seed

Low-level API allows receiving the lambda parameter, which can be used
to estimate the average payload embedded into the image


>>> alpha_hat = cl.simulate._ternary.average_payload(
...   lbda=lbda,  # lambda (optimized)
...   rho_p1=rho_p1,  # distortion of +1
...   rho_m1=rho_m1)  # distortion of -1