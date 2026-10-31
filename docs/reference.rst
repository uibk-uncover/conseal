Reference
=========

.. toctree::
   :maxdepth: 1

   reference/jpeg
   reference/spatial
   reference/simulator
   reference/coding
   reference/color
   reference/tools

The steganographic embedding in the package can be accessed on three levels of abstraction.
Higher levels need less code, lower levels give more control.
For an explanation, see :doc:`guide/levels` in the user guide.


High-level API
--------------

On high-level API, the embedding is a black-box.
You pass in the cover image and obtain the stego image,
see the simulators of the :doc:`JPEG <reference/jpeg>` and :doc:`spatial <reference/spatial>` methods.

.. code-block:: python

   jpeg.Y = cl.uerd.simulate_single_channel(
       y0=jpeg.Y,      # quantized cover DCT
       qt=jpeg.qt[0],  # quantization table
       alpha=0.4,      # embedding rate
       seed=12345,     # seed for PRNG
   )


Mid-level API
-------------

On mid-level API, you separately compute the cost (automatically adjusted for wet elements already), and run the :doc:`simulator <reference/simulator>`.
Unlike the high-level API, the simulator returns only the embedding changes, which are to be added to the cover.

.. code-block:: python

   rhos = cl.uerd.compute_cost_adjusted(
       y0=jpeg.Y,      # DCT
       qt=jpeg.qt[0],  # QT
   )
   jpeg.Y += cl.simulate(
       rhos=rhos,                # costs of +1 and -1 changes
       alpha=0.4,                # embedding rate
       n=cl.tools.nzAC(jpeg.Y),  # cover size
       seed=12345,               # seed for PRNG
   )


Low-level API
-------------

On the low-level API, you can access
the raw cost (without wet-pixel adjustment),
the probability calculation (together with the lambda parameter), and
the simulator which takes the probabilities.

.. code-block:: python

   rho = cl.uerd._costmap.compute_cost(
       y0=jpeg.Y,      # DCT
       qt=jpeg.qt[0],  # QT
   )
   # ... (sanitize rho, create rho_p1 and rho_m1)
   (p_p1, p_m1), lbda = cl.simulator.probability(
       rhos=(rho_p1, rho_m1),    # embedding costs
       alpha=0.4,                # embedding rate
       n=cl.tools.nzAC(jpeg.Y),  # cover size
   )
   jpeg.Y += cl.simulator.sample(
       (p_p1, p_m1),  # probabilities of +1 and -1 changes
       seed=12345,    # seed for PRNG
   )

Some methods, such as nsF5 and LSB, give the probabilities directly.

.. code-block:: python

   (p_p1, p_m1), _ = cl.nsF5._costmap.probability(
       y0=jpeg.Y,  # DCT
       alpha=0.4,  # embedding rate
   )
