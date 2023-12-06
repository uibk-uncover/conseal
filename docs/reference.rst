Reference
=========

.. contents:: Table of Contents
   :local:


High-level API
--------------

Simulators of JPEG Steganography
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

J-UNIWARD
"""""""""

.. autofunction:: conseal.juniward.simulate_single_channel

.. autoclass:: conseal.juniward.Implementation
   :members: JUNIWARD_ORIGINAL, JUNIWARD_FIX_OFF_BY_ONE


UERD
""""

.. autofunction:: conseal.uerd.simulate_single_channel


nsF5
""""

.. autofunction:: conseal.nsF5.simulate_single_channel


Mid-level API
-------------

Distortion
^^^^^^^^^^

J-UNIWARD
"""""""""

.. autofunction:: conseal.juniward.compute_distortion


UERD
""""

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

Cost or probability
^^^^^^^^^^^^^^^^^^^

J-UNIWARD
"""""""""

.. autofunction:: conseal.juniward._costmap.compute_cost


UERD
""""

.. autofunction:: conseal.uerd._costmap.compute_cost

nsF5
""""

.. autofunction:: conseal.nsF5._costmap.probability

Low-level Simulator API
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: conseal.simulate._ternary.probability

.. autofunction:: conseal.simulate._ternary.simulate