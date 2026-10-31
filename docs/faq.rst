FAQ
===

General
-------

What is conseal for?
^^^^^^^^^^^^^^^^^^^^

conseal implements modern image steganography for research,
mostly to generate stego images for steganalysis.
It covers the cost functions of the methods,
and simulates the embedding changes, see :doc:`guide/simulator`.

Can conseal embed and extract real messages?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Partly. Most methods only simulate the embedding,
which is what steganalysis research needs.
For nsF5, conseal also embeds and extracts real messages with wet-paper codes,
e.g., to study the loss of practical coding against the simulator.

conseal is not a stego system for real-world use, though.
It has no encryption, no message framing, and no key management,
and its files may differ from the files of cameras or common software,
see the warning in :doc:`guide/loading`.
We do not plan to turn it into one.

Why simulate instead of embedding a message?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modern steganography separates the costs from the coding.
Practical codes, such as syndrome-trellis codes, embed close to the optimum,
which the simulator assumes.
Simulation is faster and simpler, and its changes are close to those of practical codes,
so it is the standard in steganalysis research.
To simulate a practical code, see the coding efficiency in :doc:`guide/simulator`.

Does conseal do steganalysis?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No, for steganalysis features and detectors,
see our sister project `sealwatch <https://sealwatch.readthedocs.io/>`__.

How do I cite conseal?
^^^^^^^^^^^^^^^^^^^^^^

There is no paper on conseal yet.
Please cite the `repository <https://github.com/uibk-uncover/conseal>`__,
with the version you used, which you get as follows.

.. code-block:: python

   cl.__version__


Using conseal
-------------

Which unit is the embedding rate in?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bits per pixel for spatial methods,
and bits per nonzero AC coefficient for JPEG methods, see :doc:`guide/payload`.

Why does my stego differ from the original Matlab implementation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, conseal draws its random numbers as numpy does,
and some methods default to an implementation with fixed bugs.
Both can be switched to match the original,
see :doc:`guide/reproducibility`.
The default random numbers do not change the security of the methods.

Why does pip try to compile Rust?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no prebuilt wheel for your platform,
so pip builds conseal from source, which needs the Rust toolchain,
see :doc:`guide/backends`.

Why do I get a warning about falling back to Python?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Rust backend needs certain input types, e.g., a uint8 cover.
With other inputs, conseal uses Python instead, with the same result,
see :doc:`guide/backends`.

I get a deprecation warning for ``cl.simulate.ternary``. What should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The former simulator is deprecated, use ``cl.simulate(...)`` instead,
or ``cl.simulator.probability`` and ``cl.simulator.sample``,
see :doc:`guide/simulator`.


Contributing
------------

Are you open for contributions?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Absolutely. You are highly encouraged to contribute implementations of steganographic methods.
To ensure quality, we ask you to

- contribute only concepts supported by peer-reviewed research,
- document your code sufficiently, using docstrings and comments, and
- cover your contribution with unit tests,
  comparing the outputs to an existing reference implementation.

For the development setup, the conventions, and how to add a method,
see the `contributing guide <https://github.com/uibk-uncover/conseal/blob/main/CONTRIBUTING.md>`__.

Where do I report a bug?
^^^^^^^^^^^^^^^^^^^^^^^^

Open an issue on `GitHub <https://github.com/uibk-uncover/conseal/issues>`__.
