Reference
=========

The steganographic embedding in the package can be accessed on three levels of abstraction.
This creates a threshold between simplicity and flexibility.
For more information, see the `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#steganographic-design>`__.

.. contents:: Table of Contents
   :local:
   :depth: 2



High-level API
--------------

On high-level API, the embedding is a black-box.
You pass in the cover image and obtain the stego image.

Currently, there are five steganography simulators implemented: LSB, J-UNIWARD, UERD, nsF5 and EBS.


Simulators of JPEG Steganography
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The JPEG steganography operates on DCT coefficients.
To learn on how to acquire them, see the `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#jpeg-and-dct>`__.

J-UNIWARD simulator
"""""""""""""""""""

.. autofunction:: conseal.lsb.simulate

.. autoclass:: conseal.lsb.Change
   :members: LSB_REPLACEMENT, LSB_MATCHING

J-UNIWARD simulator
"""""""""""""""""""

.. autofunction:: conseal.juniward.simulate_single_channel

.. autoclass:: conseal.juniward.Implementation
   :members: JUNIWARD_ORIGINAL, JUNIWARD_FIX_OFF_BY_ONE

UERD simulator
""""""""""""""

.. autofunction:: conseal.uerd.simulate_single_channel

nsF5 simulator
""""""""""""""

.. autofunction:: conseal.nsF5.simulate_single_channel

EBS simulator
""""""""""""""

.. autofunction:: conseal.ebs.simulate_single_channel

.. autoclass:: conseal.ebs.Implementation
   :members: EBS_ORIGINAL, EBS_FIX_WET


Mid-level API
-------------

On mid-level API, you separately compute the cost (automatically adjusted for wet elements already), and run the simulator.

Adjusted cost
^^^^^^^^^^^^^

J-UNIWARD adjusted cost
"""""""""""""""""""""""

.. autofunction:: conseal.juniward.compute_cost_adjusted

UERD adjusted cost
""""""""""""""""""

.. autofunction:: conseal.uerd.compute_cost_adjusted

EBS adjusted cost
"""""""""""""""""

.. autofunction:: conseal.ebs.compute_cost_adjusted

LSB adjusted cost
"""""""""""""""""

.. autofunction:: conseal.lsb.compute_cost_adjusted


Mid-level Simulator API
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conseal.simulate.ternary

Other utilities
^^^^^^^^^^^^^^^

.. autofunction:: conseal.tools.nzAC

.. autofunction:: conseal.tools.AC


Low-level API
-------------

On the low-level API, you can access
the raw cost (without wet-pixel adjustment),
the probability calculation (together with the lambda parameter), and
the simulator which takes the probabilities.

Raw cost
^^^^^^^^

J-UNIWARD cost
""""""""""""""

.. autofunction:: conseal.juniward._costmap.compute_cost


UERD cost
"""""""""

.. autofunction:: conseal.uerd._costmap.compute_cost


EBS cost
""""""""

.. autofunction:: conseal.ebs._costmap.compute_cost

LSB cost
""""""""

.. autofunction:: conseal.lsb._costmap.compute_cost


Probability
^^^^^^^^^^^

nsF5 probability
""""""""""""""""

.. autofunction:: conseal.nsF5._costmap.probability

LSB probability
""""""""""""""""

.. autofunction:: conseal.lsb._costmap.probability


Low-level Simulator API
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conseal.simulate._ternary.probability

.. autofunction:: conseal.simulate._ternary.simulate