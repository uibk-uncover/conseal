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

Currently, there are seven steganography simulators implemented: EBS, LSB, HUGO, HILL, J-UNIWARD, nsF5 and UERD.


Simulators of Spatial Steganography
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The spatial steganography operates on pixels.

HILL simulator
"""""""""""""""

.. autofunction:: conseal.hill.simulate_single_channel

HUGO simulator
""""""""""""""

.. autofunction:: conseal.hugo.simulate_single_channel

LSB simulator
"""""""""""""

.. autofunction:: conseal.lsb.simulate

.. autoclass:: conseal.lsb.Change
   :members: LSB_REPLACEMENT, LSB_MATCHING

MiPOD simulator
"""""""""""""""""""

.. autofunction:: conseal.mipod.simulate_single_channel

.. autoclass:: conseal.mipod.Implementation
   :members: MiPOD_ORIGINAL, MiPOD_FIX_WET

S-UNIWARD simulator
"""""""""""""""""""

.. autofunction:: conseal.suniward.simulate_single_channel

WOW simulator
"""""""""""""""""""

.. autofunction:: conseal.wow.simulate_single_channel


Simulators of JPEG Steganography
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The JPEG steganography operates on DCT coefficients.
To learn on how to acquire them, see the `glossary <https://conseal.readthedocs.io/en/latest/glossary.html#jpeg-and-dct>`__.


EBS simulator
"""""""""""""

.. autofunction:: conseal.ebs.simulate_single_channel

.. autoclass:: conseal.ebs.Implementation
   :members: EBS_ORIGINAL, EBS_FIX_WET

F5 simulator
""""""""""""

.. autofunction:: conseal.F5.simulate_single_channel


J-UNIWARD simulator
"""""""""""""""""""

.. autofunction:: conseal.juniward.simulate_single_channel

.. autoclass:: conseal.juniward.Implementation
   :members: JUNIWARD_ORIGINAL, JUNIWARD_FIX_OFF_BY_ONE

nsF5 simulator
""""""""""""""

.. autofunction:: conseal.nsF5.simulate_single_channel

UERD simulator
""""""""""""""

.. autofunction:: conseal.uerd.simulate_single_channel


Mid-level API
-------------

On mid-level API, you separately compute the cost (automatically adjusted for wet elements already), and run the simulator.

Adjusted cost
^^^^^^^^^^^^^


EBS adjusted cost
"""""""""""""""""

.. autofunction:: conseal.ebs.compute_cost_adjusted

HILL adjusted cost
""""""""""""""""""

.. autofunction:: conseal.hill.compute_cost_adjusted

HUGO adjusted cost
""""""""""""""""""

.. autofunction:: conseal.hugo.compute_cost_adjusted

J-UNIWARD adjusted cost
"""""""""""""""""""""""

.. autofunction:: conseal.juniward.compute_cost_adjusted

LSB adjusted cost
"""""""""""""""""

.. autofunction:: conseal.lsb.compute_cost_adjusted

nsF5 adjusted cost
""""""""""""""""""

.. autofunction:: conseal.nsF5.compute_cost_adjusted

S-UNIWARD adjusted cost
"""""""""""""""""""""""

.. autofunction:: conseal.suniward.compute_cost_adjusted

UERD adjusted cost
""""""""""""""""""

.. autofunction:: conseal.uerd.compute_cost_adjusted

WOW adjusted cost
"""""""""""""""""

.. autofunction:: conseal.wow.compute_cost_adjusted


Mid-level Simulator API
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conseal.simulate.ternary

.. autofunction:: conseal.simulate.binary

Steganographic coding
^^^^^^^^^^^^^^^^^^^^^

The simulator can be modified by incorporating coding-specific features.

.. autofunction:: conseal.coding.efficiency

Hamming codes
"""""""""""""

.. autofunction:: conseal.coding.hamming.efficiency

Wet-paper codes
"""""""""""""""

.. autofunction:: conseal.coding.hamming.generate_H


Other utilities
^^^^^^^^^^^^^^^

.. autofunction:: conseal.tools.nzAC

.. autofunction:: conseal.tools.AC

.. autofunction:: conseal.tools.permute.password_to_seed


Low-level API
-------------

On the low-level API, you can access
the raw cost (without wet-pixel adjustment),
the probability calculation (together with the lambda parameter), and
the simulator which takes the probabilities.

Raw cost
^^^^^^^^


EBS cost
""""""""

.. autofunction:: conseal.ebs._costmap.compute_cost

HILL cost
"""""""""

.. autofunction:: conseal.hill._costmap.compute_cost

HUGO cost
"""""""""

.. autofunction:: conseal.hugo._costmap.compute_cost

J-UNIWARD cost
""""""""""""""

.. autofunction:: conseal.juniward._costmap.compute_cost

LSB cost
""""""""

.. autofunction:: conseal.lsb._costmap.compute_cost

nsF5 cost
"""""""""

.. autofunction:: conseal.nsF5._costmap.compute_cost

S-UNIWARD cost
""""""""""""""

.. autofunction:: conseal.suniward._costmap.compute_cost

UERD cost
"""""""""

.. autofunction:: conseal.uerd._costmap.compute_cost

WOW cost
""""""""

.. autofunction:: conseal.wow._costmap.compute_cost


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