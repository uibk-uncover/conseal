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
wet-paper codes, LDGM codes, Hamming codes, etc.
Researchers can skip the coding part by simulating it.
The advantages of simulation over true message coding are

- faster runtime,
- easier implementation,
- ethical code publishing.

The simplest simulator assumes mutual independence (MI) between elements.
For each element of index $i$, MI simulator performs two steps,

- converting distortion :math:`\rho_i` into probability :math:`p_i`; and
- simulating the embedding change from Bernoulli or Multinoulli distribution,
parameterized by the probability :math:`p_i`.

The probability conversion is done using Boltzmann-Gibbs distribution,
.. The Boltzmann distribution maximizes the entropy at given energy,
.. which translates to maximal message size at given distortion in steganography

.. math::
   p_i^{(v)} = 1 / Z * exp( - \lambda \rho_i^{(v)}),

where p_i^{(v)} is the probability of change :math:`v`,
:math:`\rho_i^{(v)}` is the distortion associated with the change :math:`v`,
:math:`\lambda` is the parameter encorporating required message size,
and :math:`Z` is a normalization constant.

For the common case of ternary embedding with
$+1$ and $-1$ changes, and
:math:`\rho^{(0)}=0`,
which is also implemented in the `conseal` package,
the equation above can be written as follows.

.. math::
   p_i^{(+1)} &= \frac{exp( - \lambda \rho_i^{(+1)})}{1+exp(-\lambda \rho_i^{(+1)})+exp(-\lambda \rho_i^{(-1)})} \\
   p_i^{(-1)} &= \frac{exp( - \lambda \rho_i^{(-1)})}{1+exp(-\lambda \rho_i^{(+1)})+exp(-\lambda \rho_i^{(-1)})} \\
