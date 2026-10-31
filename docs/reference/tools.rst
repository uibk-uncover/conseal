Tools
=====

Helpers shared by the methods, also useful on their own.


JPEG and DCT
------------

Counting the coefficients, e.g., for the payload in bits per nonzero AC coefficient, see :doc:`../guide/payload`.

.. autofunction:: conseal.tools.nzAC

.. autofunction:: conseal.tools.AC

Converting between the DCT layouts of jpeglib and jpegio, see :doc:`../guide/loading`.

.. autofunction:: conseal.tools.jpeglib_to_jpegio

.. autofunction:: conseal.tools.jpegio_to_jpeglib

Decompressing and transforming.

.. autofunction:: conseal.tools.decompress_channel

.. autofunction:: conseal.tools.dct.dct2

.. autofunction:: conseal.tools.dct.idct2

.. autofunction:: conseal.tools.dct.block_dct2

.. autofunction:: conseal.tools.dct.block_idct2

.. autofunction:: conseal.tools.dct.compute_DCT_mat

.. autofunction:: conseal.tools.dct.compute_dct_basis_functions


Spatial
-------

.. autofunction:: conseal.tools.spatial.daubechies8


Entropy
-------

.. autofunction:: conseal.tools.entropy

.. autofunction:: conseal.tools.inv_entropy


Order and seeds
---------------

Seeds per image, see :doc:`../guide/dataset`, and the order of the elements.

.. autofunction:: conseal.tools.password_to_seed

.. autofunction:: conseal.tools.iterate

.. autofunction:: conseal.tools.iterate_ac

.. autofunction:: conseal.tools.blocks


Backends
--------

Switching between the Rust and the Python implementation, see :doc:`../guide/backends`.

.. autoclass:: conseal.tools.Backend
   :members: BACKEND_RUST, BACKEND_PYTHON

.. autofunction:: conseal.tools.get_backend

.. autofunction:: conseal.tools.set_backend
