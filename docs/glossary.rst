Glossary
===================================

This part briefly explains the concepts behind the library.
The rest of documentation will refer to this page as a knowledge base.

.. contents:: Table of Contents
   :local:
   :depth: 1

Embedding simulation
--------------------

Modern steganography consists of distortion calculation,
followed by coding, e.g., using syndrome-trellis codes (STC),
low-density generator-matrix (LDGM), wet-paper codes, codes, Hamming codes, etc.
Researchers can skip the coding part by simulating it.
The advantages of simulation over true message coding are

- faster runtime,
- easier implementation, and
- ethical code publishing.

Mutually independent (MI) simulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple simulator assumes mutual independence (MI) between elements.
For each element of index :math:`i`, MI simulator performs two steps,

- converting distortion :math:`\rho_i` into probability :math:`p_i`; and
- simulating the embedding change from Bernoulli or Multinoulli distribution, parameterized by the probability :math:`p_i`.

The probability conversion is done using Boltzmann-Gibbs distribution,

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
   p_i^{(+1)} &= \frac{\text{exp}( - \lambda \rho_i^{(+1)})}{1+\text{exp}(-\lambda \rho_i^{(+1)})+exp(-\lambda \\text{rho}_i^{(-1)})} \\
   p_i^{(-1)} &= \frac{\text{exp}( - \lambda \rho_i^{(-1)})}{1+\text{exp}(-\lambda \rho_i^{(+1)})+exp(-\lambda \\text{rho}_i^{(-1)})} \\


Sublattice simulator
^^^^^^^^^^^^^^^^^^^^

*Will be added in the future*


Distortion functions
--------------------

Uniform Embedding Revisited Distortion (UERD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*TODO: To be completed*

JPEG Universal Wavelet Relative Distortion (J-UNIWARD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*TODO: To be completed*

Non-shrinkage F5 (nsF5)
^^^^^^^^^^^^^^^^^^^^^^^

*TODO: To be completed*
