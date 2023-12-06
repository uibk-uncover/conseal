Reference
=========

The steganographic embedding in the package can be controlled on three levels of abstraction.
This creates a threshold between breivity and flexibility.

.. contents:: Table of Contents
   :local:
   :maxdepth: 2


High-level API
--------------

Simulators of JPEG Steganography
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Mid-level API
-------------

Distortion
^^^^^^^^^^

J-UNIWARD distortion
""""""""""""""""""""

.. autofunction:: conseal.juniward.compute_distortion


UERD distortion
"""""""""""""""

.. autofunction:: conseal.uerd.compute_distortion

Mid-level Simulator API
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conseal.simulate.ternary

Tools
^^^^^

.. autofunction:: conseal.tools.nzAC

.. autofunction:: conseal.tools.AC


Low-level API
-------------

Cost or probability (no wet-cost adjustement)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

J-UNIWARD cost
""""""""""""""

.. autofunction:: conseal.juniward._costmap.compute_cost


UERD cost
"""""""""

.. autofunction:: conseal.uerd._costmap.compute_cost

nsF5 probability
""""""""""""""""

.. autofunction:: conseal.nsF5._costmap.probability

Low-level Simulator API
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conseal.simulate._ternary.probability

.. autofunction:: conseal.simulate._ternary.simulate