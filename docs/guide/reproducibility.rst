Reproducibility
===============

conseal reproduces the reference implementations of the methods exactly,
down to the stego image, given the same seed.
This page explains the settings that control it:
the original implementations and their deviations,
and the random number generator of the simulator.


Original and fixed implementations
----------------------------------

Some reference implementations contain bugs.
For these methods, conseal offers the original implementation,
which reproduces the reference including the bug,
and a fixed one.
Use the original to reproduce published results,
and the fixed one for new experiments.

.. list-table::
   :class: new-tab
   :widths: 15 35 50
   :header-rows: 1

   * - Method
     - Original
     - Fixed
   * - EBS
     - ``cl.EBS_ORIGINAL``
     - ``cl.EBS_FIX_WET``, handles wet elements correctly
   * - J-UNIWARD
     - ``cl.JUNIWARD_ORIGINAL``
     - ``cl.JUNIWARD_FIX_OFF_BY_ONE``, fixes an off-by-one error in the cost,
       see `Lorch, Off-By-One Implementation Error in J-UNIWARD <https://arxiv.org/abs/2305.19776>`__
   * - MiPOD
     - ``cl.MiPOD_ORIGINAL``
     - ``cl.MiPOD_FIX_WET``, handles wet elements correctly

Choose it with the ``implementation`` parameter.

.. code-block:: python

   jpeg.Y = cl.juniward.simulate_single_channel(
       y0=jpeg.Y,
       qt=jpeg.qt[0],
       x0=im0.spatial[..., 0],
       alpha=0.4,
       implementation=cl.JUNIWARD_ORIGINAL,  # reproduce the reference
       seed=12345,
   )

Mind the defaults.
J-UNIWARD and MiPOD use the fixed implementation by default,
while the EBS simulator uses the original one.


Random number generator
-----------------------

The simulator draws one uniform random number per element,
and changes the element if the number falls below its change probability.
To get the same stego as a reference implementation,
the simulator must draw the same numbers, in the same order.

The ``generator`` parameter chooses the random number generator.

- ``None`` (default for most methods) is the numpy default generator.
  It is reproducible within conseal, given the seed,
  but it differs from other implementations.
- ``'MT19937'`` is the Mersenne Twister, seeded as in Matlab,
  i.e., ``rng(seed)`` or ``RandStream('mt19937ar', 'Seed', seed)``,
  and as ``np.random.seed(seed)`` in numpy.
- A callable ``generator(shape)`` returning uniform numbers of the given shape,
  for any other generator.

The ``order`` parameter assigns the random numbers to the elements.

- ``'C'`` (default) is row-wise, as numpy fills arrays,
  e.g., in Python reference implementations.
- ``'F'`` is column-wise, as Matlab fills arrays with ``rand(size)``.

.. code-block:: python

   # as the Matlab reference by DDE Lab
   x1 = cl.suniward.simulate_single_channel(
       x0=x0,
       alpha=0.4,
       generator='MT19937',  # Mersenne Twister, as Matlab
       order='F',            # column-wise, as Matlab
       seed=12345,
   )

The defaults follow Python, unlike most references.
This does not affect security:
the change probabilities stay the same, only the drawn numbers differ.

The JPEG simulators arrange the coefficients as a 2D image before sampling,
as Matlab reads them, so the order of the elements matches too.
UERD is the exception, it samples on the 8x8 blocks,
so it matches its reference only on the mid level, see below.

The search for λ, which sets the payload,
follows the binary search of the DDE Lab by default,
including the rounding of the payload to whole bits.


Settings per method
-------------------

The settings below reproduce the reference implementations,
as tested in the unit tests, see the `methods <../index.html#methods>`__.

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Method
     - Reference
     - Settings
   * - EBS
     - Troyes
     - default, i.e., ``implementation=cl.EBS_ORIGINAL, generator='MT19937', order='C'``
   * - J-MiPOD
     - Troyes
     - default
   * - J-UNIWARD
     - Binghamton
     - ``implementation=cl.JUNIWARD_ORIGINAL, generator='MT19937'``
   * - nsF5
     - Binghamton
     - default
   * - UERD
     - Troyes
     - mid level, ``cl.simulate.sample(ps, generator='MT19937', seed=seed)``
       on the costs arranged by ``cl.tools.jpeglib_to_jpegio``
   * - HILL
     - Barcelona
     - ``generator='MT19937'``
   * - HUGO
     - Binghamton
     - ``generator='MT19937', order='F'``
   * - MiPOD
     - Binghamton
     - ``implementation=cl.MiPOD_ORIGINAL``
   * - S-UNIWARD
     - Binghamton
     - ``generator='MT19937', order='F'``
   * - WOW
     - Binghamton
     - ``generator='MT19937', order='F'``

J-MiPOD, nsF5 and MiPOD draw their random numbers as their Matlab reference,
regardless of the settings.
F5, S-UNIGARD and LSB have no reference implementation to reproduce.
