Backends
========

Some cost functions have two implementations,
in Python and in Rust.
Rust is the default, and both give the same costs,
up to floating-point rounding, as tested in the unit tests.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Rust implementation
     - Requirements
   * - HILL
     - uint8 cover
   * - J-UNIWARD
     - fixed implementation, uint8 decompressed cover, uint16 quantization table,
       separable filters
   * - S-UNIWARD
     - uint8 cover
   * - UERD
     - int16 DCT and uint16 quantization table, as read by jpeglib
   * - WOW
     - uint8 cover, separable filters
   * - Wet-paper codes
     - fewer than 256 symbols

The other methods are implemented in Python only.
If the input does not meet the requirements,
conseal falls back to Python with a warning.
The original J-UNIWARD falls back silently,
since it exists only in Python.


Switching the backend
---------------------

Switch the backend globally,

.. code-block:: python

   cl.set_backend(cl.BACKEND_PYTHON)

or only for a block of code.

.. code-block:: python

   with cl.BACKEND_PYTHON:
       rhos = cl.hill.compute_cost_adjusted(x0)


Installation
------------

The Rust backend comes prebuilt for all major platforms.
``pip install conseal`` downloads a Python wheel with the compiled Rust code,
so no Rust is needed to use it.
One wheel per platform serves all supported Python versions.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Platform
     - Architecture
   * - Linux
     - x86-64
   * - macOS
     - Apple Silicon (arm64)
   * - Windows
     - x86-64

On other platforms, pip builds the package from source.
This requires the `Rust toolchain <https://rustup.rs/>`__ to be installed,

.. code-block:: console

   $ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   $ pip install conseal

and the same holds for installing from a clone of the repository.

.. code-block:: console

   $ git clone https://github.com/uibk-uncover/conseal
   $ cd conseal
   $ pip install .
