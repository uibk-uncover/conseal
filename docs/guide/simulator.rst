Simulator
=========

The simulator turns costs into embedding changes, without embedding a message.
It changes each element with probability p ~ exp(-λρ),
i.e., cheap elements more often,
and searches λ, s.t., the probabilities carry the payload.
Then it samples the changes.

.. code-block:: python

   ps, lbda = cl.simulator.probability(rhos, alpha=0.4)  # search λ
   delta = cl.simulator.sample(ps, seed=12345)           # sample changes

``cl.simulate`` does both in one call.


Solvers
-------

The search for λ is done by one of four solvers.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Solver
     - Use
   * - ``cl.simulator.SOLVER_BSEARCH_DDE``
     - default, binary search of the DDE Lab,
       reproduces their Matlab implementations
   * - ``cl.simulator.SOLVER_NEWTON``
     - Newton's method, fastest and most precise
   * - ``cl.simulator.SOLVER_BRENT``
     - Brent's method, as fast, without the derivative,
       e.g., for a custom objective
   * - ``cl.simulator.SOLVER_BSEARCH``
     - plain binary search, robust but slow

The DDE solver stops at a payload within 1/1000 of the target,
by design of the original.
For new experiments, Newton's method is about twice as fast,
and hits the payload a few hundred times more precisely.

.. code-block:: python

   ps, lbda = cl.simulator.probability(
       rhos,
       alpha=0.4,
       solver=cl.simulator.SOLVER_NEWTON,
   )


Coding efficiency
-----------------

By default, the simulator assumes optimal coding, which implies :math:`e = \alpha / H^{-1}(\alpha)`.

.. code-block:: python

   # optimal coding
   ps, lbda = cl.simulator.probability(rhos, alpha=0.4)

Practical steganography needs more changes for the same payload.
For example, LSB steganography embeds at a constant embedding efficiency e=2 bits per change.


.. code-block:: python

   # 2 bits per change
   ps, lbda = cl.simulator.probability(rhos, alpha=0.4, e=2)


The efficiency parameter can be used to impose the coding loss of a specific coding method.
For instance, for Hamming codes, it can be done as follows.

.. code-block:: python

   # coding loss of Hamming codes
   e = cl.coding.hamming.efficiency(alpha=0.4)
   ps, lbda = cl.simulator.probability(rhos, alpha=0.4, e=e)


Changes
-------

One cost tensor per change gives binary, ternary, or more changes.
By default, two tensors are +1 and -1 changes,
other changes are given by ``values``.

.. code-block:: python

   # pentary, +-1 and +-2 changes
   ps, lbda = cl.simulator.probability((rho_p1, rho_m1, rho_p2, rho_m2), alpha=0.4)
   delta = cl.simulator.sample(ps, values=(1, -1, 2, -2), seed=12345)


No-change cost
--------------

By default, not changing an element costs nothing,
and the no-change state is implicit.
If it has a cost, e.g., in side-informed embedding,
give it as a state of its own, with value 0,
and set the number of states q to the number of tensors.

.. code-block:: python

   ps, lbda = cl.simulator.probability((rho_p1, rho_m1, rho_0), alpha=0.4, q=3)
   delta = cl.simulator.sample(ps, values=(1, -1, 0), seed=12345)

For a payload-limited sender, this is the same as costs relative to no change,
since only the differences of the costs matter.

.. code-block:: python

   ps, lbda = cl.simulator.probability((rho_p1 - rho_0, rho_m1 - rho_0), alpha=0.4)

With the no-change state last, as above, both give the same changes.
For a distortion-limited sender, they differ,
since the shift also changes the distortion.


Legacy interface
----------------

The former simulator, ``cl.simulate.binary``, ``cl.simulate.ternary``,
``cl.simulate._binary`` and ``cl.simulate._ternary``, is deprecated.
It still works, with a ``DeprecationWarning``,
and gives the same changes as the new interface.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Legacy
     - Replacement
   * - ``cl.simulate.binary``, ``cl.simulate.ternary``
     - ``cl.simulate``
   * - ``cl.simulate._binary.probability``, ``cl.simulate._ternary.probability``
     - ``cl.simulator.probability``
   * - ``cl.simulate._binary.simulate``, ``cl.simulate._ternary.simulate``
     - ``cl.simulator.sample``
   * - ``cl.simulate.calc_lambda``
     - ``cl.simulator.search``

The other legacy functions, such as ``average_payload``,
are kept in ``cl.simulate_old`` until their replacement.
