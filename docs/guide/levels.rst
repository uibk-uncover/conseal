Levels of the API
=================

Modern steganography separates two steps:
the costs of changing each cover element,
and the embedding, which changes the cheap elements with a higher probability.
conseal exposes this separation on three levels.
Higher levels need less code, lower levels give more control.

The examples use UERD on a JPEG, loaded as in :doc:`loading`.

.. code-block:: python

   import conseal as cl
   import jpeglib

   jpeg = jpeglib.read_dct("cover.jpeg")


High level
----------

One call takes the cover and returns the stego.

.. code-block:: python

   y1 = cl.uerd.simulate_single_channel(
       y0=jpeg.Y,       # DCT of cover luminance
       qt=jpeg.qt[0],   # quantization table
       alpha=0.4,       # embedding rate
       seed=12345,      # seed for PRNG
   )


Mid level
---------

The costs and the simulation are separate calls.
The costs are already adjusted,
i.e., elements which must not change, such as saturated ones, are wet.
The simulator returns the changes, to be added to the cover.

.. code-block:: python

   # costs of +1 and -1 changes
   rhos = cl.uerd.compute_cost_adjusted(y0=jpeg.Y, qt=jpeg.qt[0])

   # simulate the changes
   delta = cl.simulate.simulate(
       rhos=rhos,
       alpha=0.4,                 # embedding rate
       n=cl.tools.nzAC(jpeg.Y),   # cover size, see Payload
       seed=12345,                # seed for PRNG
   )
   y1 = jpeg.Y + delta

This is where you can modify the costs,
e.g., to make some elements wet or avoid certain areas, before simulating.


Low level
---------

The raw costs come without the adjustment,
and the simulation splits into the probabilities and the sampling.
This gives access to the probabilities and to λ,
the parameter which sets the payload.

.. code-block:: python

   # raw cost, the same for +1 and -1 changes
   rho = cl.uerd._costmap.compute_cost(y0=jpeg.Y, qt=jpeg.qt[0])

   # adjust yourself, no change beyond the range of DCT
   wet_cost = 10**13
   rho_p1, rho_m1 = rho.copy(), rho.copy()
   rho_p1[jpeg.Y >= 1023] = wet_cost
   rho_m1[jpeg.Y <= -1023] = wet_cost

   # probabilities of +1 and -1 changes, and λ
   (p_p1, p_m1), lbda = cl.simulate.probability(
       rhos=(rho_p1, rho_m1),
       alpha=0.4,
       n=cl.tools.nzAC(jpeg.Y),
   )

   # sample the changes
   delta = cl.simulate.sample((p_p1, p_m1), seed=12345)
   y1 = jpeg.Y + delta

The probabilities are useful on their own,
e.g., as the selection channel for steganalysis,
or to compute the payload, see :doc:`payload`.
