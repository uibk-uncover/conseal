Simulator
=========

The simulator converts costs into embedding changes, see :doc:`../guide/simulator`.
It has two parts:
the solver searches λ and converts the costs into change probabilities,
and the sampler draws the changes from the probabilities.


Simulation
----------

Solver and sampler in one call, also available as ``cl.simulate(...)``.

.. autofunction:: conseal.simulator.simulate


Solver
------

.. autofunction:: conseal.simulator.probability

.. autofunction:: conseal.simulator.search

.. autofunction:: conseal.simulator.get_probability

.. autoclass:: conseal.simulator.LambdaSolver
   :members:
   :undoc-members:

.. autoclass:: conseal.simulator.Sender
   :members:
   :undoc-members:


Objectives
^^^^^^^^^^

The objective of the search, the payload of a payload-limited sender,
or the distortion of a distortion-limited sender, with its derivative and limits.

.. autoclass:: conseal.simulator.Objective
   :members:

.. autofunction:: conseal.simulator.entropy

.. autofunction:: conseal.simulator.d_entropy

.. autofunction:: conseal.simulator.entropy_limits

.. autofunction:: conseal.simulator.distortion

.. autofunction:: conseal.simulator.d_distortion

.. autofunction:: conseal.simulator.distortion_limits


Sampler
-------

.. autofunction:: conseal.simulator.sample
