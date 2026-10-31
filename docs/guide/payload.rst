Payload
=======

The payload is given as embedding rate α,
the message length relative to the size of the cover.


Units
-----

The size of the cover depends on the domain.

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Domain
     - Unit of α
     - Size of the cover n
   * - Spatial
     - bits per pixel (bpp)
     - number of pixels
   * - Spatial
     - bits per element (bpe)
     - number of elements, i.e., pixels times channels
   * - JPEG
     - bits per nonzero AC coefficient (bpnzAC)
     - number of nonzero AC coefficients
   * - JPEG
     - bits per coefficient (bpc)
     - number of DCT coefficients

For grayscale images, bits per pixel and bits per element are the same.
For example, α = 0.4 in a 512x512 grayscale image
is a message of 0.4 · 512 · 512 ≈ 105 kbits.
In a JPEG, α in bpnzAC means a shorter message than in bpc,
since only a part of the coefficients is nonzero.

The simulators of the methods count n themselves,
in bits per pixel or bits per nonzero AC coefficient.
UERD also takes ``payload_mode='bpc'``.
When you call the simulator directly, see :doc:`levels`,
the unit is set by n in the call.

.. code-block:: python

   rhos = cl.uerd.compute_cost_adjusted(y0=jpeg.Y, qt=jpeg.qt[0])

   # 0.4 bits per nonzero AC coefficient
   ps, lbda = cl.simulator.probability(rhos, alpha=0.4, n=cl.tools.nzAC(jpeg.Y))

   # 0.4 bits per coefficient
   ps, lbda = cl.simulator.probability(rhos, alpha=0.4, n=jpeg.Y.size)


Simulated payload
-----------------

The simulator does not embed a message.
It changes each element with a probability,
s.t., the entropy of the changes equals the payload.
An optimal code would embed this payload with these changes.
You can check it from the probabilities.

.. code-block:: python

   rhos = cl.hill.compute_cost_adjusted(x0)
   ps, lbda = cl.simulator.probability(rhos, alpha=0.4)

   cl.tools.entropy(*ps) / x0.size  # ≈ 0.4 bits per pixel

Practical codes need more changes than the simulator,
so the simulator is slightly optimistic about the detectability.


Payload or distortion
---------------------

By default, the sender is payload-limited:
it embeds the given payload with the least distortion.
A distortion-limited sender instead embeds as much as possible
within a given average cost per element.

.. code-block:: python

   ps, lbda = cl.simulator.probability(
       rhos,
       alpha=0.05,          # average cost per pixel
       objective=cl.DLS,    # distortion-limited sender
   )
