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
>>> im_spatial = jpeglib.read_spatial("cover.jpeg", jpeglib.JCS_GRAYSCALE)
>>> im_dct = jpeglib.read_dct("cover.jpeg")


``Conseal`` provides several API on different levels of abstraction.
The lower the level, the more control, but more verbose code becomes.


At high-level API
-----------------

Using the high-level API, you call a single function, provide the cover, and receive stego.

>>> im_dct.Y = cl.nsF5.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,
...   quantization_table=im_dct.qt[0],
...   embedding_rate=0.4,
...   seed=12345)

>>> im_dct.Y = cl.uerd.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,
...   quantization_table=im_dct.qt[0],
...   embedding_rate=0.4,
...   seed=12345)

>>> im_dct.Y = cl.juniward.simulate_single_channel(
...   cover_dct_coeffs=im_dct.Y,
...   quantization_table=im_dct.qt[0],
...   cover_spatial=im_spatial.spatial[..., 0],
...   embedding_rate=0.4,
...   seed=12345)

J-UNIWARD computes the distortion in decompressed domain,
so it must be provided alongside the DCT coefficients.


At mid-level API
----------------

Mid-level API exposes the separation principle.
It allows user to separately calculate the distortion, and perform the simulation of coding.

>>> rhoP1, rhoM1 = cl.juniward.compute_distortion(
...   cover_dct_coeffs=im_dct.Y,
...   quantization_table=im_dct.qt[0],
...   cover_spatial=im_spatial.spatial[..., 0])
>>> im_dct.Y += cl.simulate.ternary(
...   rhoP1=rhoP1,
...   rhoM1=rhoM1,
...   alpha=0.4,
...   n=im_dct.Y.size,
...   seed=12345)

>>> rhoP1, rhoM1 = cl.uerd.compute_distortion(
...   cover_dct_coeffs=im_dct.Y,
...   quantization_table=im_dct.qt[0])
>>> im_dct.Y += cl.simulate.ternary(
...   rhoP1=rhoP1,
...   rhoM1=rhoM1,
...   alpha=0.4,
...   n=im_dct.Y.size,
...   seed=12345)

Notice that unlike high-level API, mid-level and low-level API return difference tensor,
that is to be added to the cover.


At low-level API
----------------

Low-level API allows accessing the raw costs (without wet cost modification),
as well as the probabilities and simulation.


>>> rho = cl.uerd._costmap.compute_cost(
...   cover_dct_coeffs=im_dct.Y,
...   quantization_table=im_dct.qt[0])
>>> # ... (sanitize rho, create rhoP1 and rhoM1)
>>> (pP1, pM1), lbda = cl.simulate._ternary.probability(
...   rhoP1=rhoP1,
...   rhoM1=rhoM1,
...   alpha=0.4,
...   n=im_dct.Y.size)
>>> im_dct.Y += cl.simulate._ternary.simulate(
...   pChangeP1=pP1,
...   pChangeM1=pM1,
...   seed=12345)

Low-level API allows receiving the lambda parameter, which can be used
to estimate the average payload embedded into the image


>>> alpha_hat = cl.simulate._ternary.average_payload(
...   lbda=lbda,
...   rhoP1=rhoP1,
...   rhoM1=rhoM1)