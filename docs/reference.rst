Reference
=========

.. contents:: Table of Contents
   :local:


High-level API
--------------

Simulators of JPEG Steganography
""""""""""""""""""""""""""""""""

.. autofunction:: conseal.juniward.simulate_single_channel

.. autoclass:: conseal.juniward.Implementation
   :members: JUNIWARD_ORIGINAL, JUNIWARD_FIX_OFF_BY_ONE

.. autofunction:: conseal.uerd.simulate_single_channel

.. autofunction:: conseal.nsF5.simulate_single_channel

Tools
"""""

.. autofunction:: conseal.tools.nzAC


Mid-level API
-------------

Distortion of JPEG Steganography
""""""""""""""""""""""""""""""""

.. autofunction:: conseal.juniward.compute_distortion

.. autofunction:: conseal.uerd.compute_distortion

Simulator API
"""""""""""""

.. autofunction:: conseal.simulate.ternary

Low-level API
-------------

Distortion of JPEG Steganography
""""""""""""""""""""""""""""""""

.. autofunction:: conseal.juniward.compute_cost

.. autofunction:: conseal.uerd.compute_cost

.. autofunction:: conseal.nsF5.probability

Simulator API
"""""""""""""

.. autofunction:: conseal.simulate._ternary.probability

.. autofunction:: conseal.simulate._ternary.simulate