JPEG methods
============

The JPEG methods embed into the quantized DCT coefficients,
see :doc:`../guide/loading`.
Each method offers a simulator, the costs adjusted for wet elements,
and the raw costs in ``_costmap``.

EBS
---

.. automodule:: conseal.ebs
   :members:
   :imported-members:

.. autofunction:: conseal.ebs._costmap.compute_cost

F5
--

.. automodule:: conseal.F5
   :members:
   :imported-members:

J-MiPOD
-------

.. automodule:: conseal.jmipod
   :members:
   :imported-members:

J-UNIWARD
---------

.. automodule:: conseal.juniward
   :members:
   :imported-members:

.. autofunction:: conseal.juniward._costmap.compute_cost

nsF5
----

.. automodule:: conseal.nsF5
   :members:
   :imported-members:

.. autofunction:: conseal.nsF5._costmap.compute_cost

.. autofunction:: conseal.nsF5._costmap.probability

UERD
----

.. automodule:: conseal.uerd
   :members:
   :imported-members:

.. autofunction:: conseal.uerd._costmap.compute_cost

