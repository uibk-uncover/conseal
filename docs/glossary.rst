Glossary
========

This part briefly explains the concepts behind the library.
The rest of documentation will refer to this page as a knowledge base.

.. contents:: Table of Contents
   :local:
   :depth: 2

Steganographic design
---------------------

*TODO: To be completed*

Embedding simulation
--------------------

Modern steganography consists of distortion calculation,
followed by coding, e.g., using syndrome-trellis codes (STC),
low-density generator-matrix (LDGM), wet-paper codes, codes, Hamming codes, etc.
In research, the coding step is usually replaced by simulating the embedding changes.
The advantages of simulation over true message coding are

- faster runtime,
- easier implementation, and
- ethical code publishing.

Mutually independent (MI) simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple simulator assumes mutual independence (MI) between elements.
For each element of index :math:`i`, the MI simulator performs two steps,

- converting distortion :math:`\rho_i` into probability :math:`p_i`; and
- simulating the embedding change from Bernoulli or Multinoulli distribution, parameterized by the probability :math:`p_i`.

The probability conversion is done using a Boltzmann-Gibbs distribution,

.. math::
   p_i^{(v)} = 1 / Z * \text{exp}( - \lambda \rho_i^{(v)}),

where :math:`p_i^{(v)}` is the probability of change :math:`v`,
:math:`\rho_i^{(v)}` is the distortion associated with the change :math:`v`,
:math:`\lambda` is the parameter encorporating required message size,
and :math:`Z` is a normalization constant.


MI simulator for ternary embedding
""""""""""""""""""""""""""""""""""

For the common case of ternary embedding with
:math:`+1` and :math:`-1` changes, and
:math:`\rho^{(0)}=0`,
which is also implemented in the `conseal` package,
the equation above can be written as follows.

.. math::
   p_i^{(+1)} &= \frac{\text{exp}( - \lambda \rho_i^{(+1)})}{1+\text{exp}(-\lambda \rho_i^{(+1)})+\text{exp}(-\lambda \rho_i^{(-1)})} \\
   p_i^{(-1)} &= \frac{\text{exp}( - \lambda \rho_i^{(-1)})}{1+\text{exp}(-\lambda \rho_i^{(+1)})+\text{exp}(-\lambda \rho_i^{(-1)})} \\


Sublattice simulator
^^^^^^^^^^^^^^^^^^^^

*Will be added in the future*


Cost functions
--------------

Uniform Embedding Revisited Distortion (UERD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*TODO: To be completed*

JPEG Universal Wavelet Relative Distortion (J-UNIWARD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*TODO: To be completed*

Entropy Block Stego (EBS)
^^^^^^^^^^^^^^^^^^^^^^^^^

*TODO: To be completed*

EBS was used as a benchmark against the SI-UNIWARD.

Non-shrinkage F5 (nsF5)
^^^^^^^^^^^^^^^^^^^^^^^

nsF5 is not a cost function, but a complete steganographic design.
Published in 2007, it does not yet follow the `design of modern steganography <https://conseal.readthedocs.io/en/latest/glossary.html#steganographic-design>`,
but performs "*permutative straddling*", when the message is spreaded across the entire cover.

This is done using pseudo-random permutation over non-zero DCT AC coefficients,
combined with wet-paper codes (the difference from F5, which solves the shrinkage problem).

Because of this construction, module ``conseal.nsF5`` cannot define function
``compute_cost_adjusted`` and ``compute_cost``, and the mid-level and low-level call sequence
differs from the other modules implementing JPEG steganography.


JPEG and DCT
------------

JPEG is a lossy image format.
Because of its popularity, it is a good target for steganography, typically done on top of DCT coefficients.
To read coefficients from JPEG and write them back, we encourage you to use our other project, `jpeglib <https://pypi.org/project/jpeglib/>`__.
In its `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html>`__, you can find a lot of details on JPEG specifically.
