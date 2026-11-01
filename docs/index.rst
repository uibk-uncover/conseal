:html_theme.sidebar_secondary.remove:

.. rst-class:: hero-title

conseal
=======

.. image:: static/seal.png
   :width: 120px
   :class: hero-logo
   :alt: conseal

.. rst-class:: hero-tagline

Making research in steganography reproducible and accessible.

**conseal** is a Python package implementing the cost functions and embedding simulators
of modern steganographic methods, for JPEG and spatial images.

.. code-block:: console

   $ pip3 install conseal

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: User guide
      :link: guide/index
      :link-type: doc

      Load a cover image, set the payload, and use the API on three levels.

   .. grid-item-card:: Reference
      :link: reference
      :link-type: doc

      Every simulator, cost function and coding method, with parameters and examples.

   .. grid-item-card:: Glossary
      :link: glossary
      :link-type: doc

      Short definitions of the terms.

   .. grid-item-card:: FAQ
      :link: faq
      :link-type: doc

      Contributing, scope of the package, and common questions.


Quick start
-----------

.. tab-set::

   .. tab-item:: JPEG

      .. code-block:: python

         import conseal as cl
         import jpeglib

         # load cover
         im0 = jpeglib.read_spatial("cover.jpeg", jpeglib.JCS_GRAYSCALE)
         jpeg = jpeglib.read_dct("cover.jpeg")

         # simulate J-UNIWARD at 0.4 bits per nonzero AC coefficient
         jpeg.Y = cl.juniward.simulate_single_channel(
             y0=jpeg.Y,               # DCT of cover luminance
             qt=jpeg.qt[0],           # quantization table
             x0=im0.spatial[..., 0],  # decompressed cover luminance
             alpha=0.4,               # embedding rate
             seed=12345,              # seed for PRNG
         )

         # save stego
         jpeg.write_dct("stego.jpeg")

   .. tab-item:: Spatial

      .. code-block:: python

         import conseal as cl
         import numpy as np
         from PIL import Image

         # load cover
         x0 = np.array(Image.open("cover.png"))

         # simulate HILL at 0.4 bits per pixel
         x1 = cl.hill.simulate_single_channel(
             x0=x0,       # cover pixels
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )

         # save stego
         Image.fromarray(x1).save("stego.png")

For other methods, swap the call for an example from the table below.


Methods
-------

.. role:: raw-html(raw)
   :format: html

.. list-table::
   :class: new-tab method-table
   :widths: 20 18 18 29 15
   :header-rows: 1

   * - Method
     - Domain
     - Usage
     - Verified
     - Paper
   * - :abbr:`EBS (Entropy block steganography)`
     - JPEG
     -
     - :verified:`troyes test/test_ebs.py::TestEBS::test_stego_reference`
     - `link <https://doi.org/10.1109/ICASSP.2012.6288246>`__
   * - F5
     - JPEG
     -
     - —
     - `link <https://doi.org/10.1007/3-540-45496-9_21>`__
   * - :abbr:`J-MiPOD (JPEG minimizing the power of optimal detector)`
     - JPEG
     -
     - :verified:`troyes test/test_jmipod.py::TestJMiPOD::test_stego_reference`
     - `link <https://doi.org/10.1145/3369412.3395075>`__
   * - :abbr:`J-UNIWARD (JPEG universal wavelet relative distortion)`
     - JPEG
     -
     - :verified:`binghamton test/test_juniward.py::TestJUNIWARD::test_stego_reference`
     - `link <https://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf>`__
   * - :abbr:`nsF5 (No-shrinkage F5)`
     - JPEG
     -
     - :verified:`binghamton test/test_nsF5.py::TestnsF5::test_stego_reference`
     - `link <https://dde.binghamton.edu/kodovsky/pdf/Fri07-ACM.pdf>`__
   * - :abbr:`UERD (Uniform embedding revisited distortion)`
     - JPEG
     -
     - :verified:`troyes test/test_uerd.py::TestUERD::test_stego_reference`
     - `link <https://doi.org/10.1109/TIFS.2015.2473815>`__
   * - :abbr:`HILL (High-low-low)`
     - Spatial
     -
     - :verified:`barcelona test/test_hill.py::TestHILL::test_stego_reference`
     - `link <https://doi.org/10.1109/ICIP.2014.7025854>`__
   * - :abbr:`HUGO (Highly undetectable stego)`
     - Spatial
     -
     - :verified:`binghamton test/test_hugo.py::TestHUGO::test_stego_reference`
     - `link <https://doi.org/10.1007/978-3-642-16435-4_13>`__
   * - :abbr:`MiPOD (Minimizing the power of optimal detector)`
     - Spatial
     -
     - :verified:`binghamton test/test_mipod.py::TestMiPOD::test_stego_reference`
     - `link <https://dde.binghamton.edu/vsedighi/pdf/TIFS2015_Content_Adaptive_Steganography_by_Minimizing_Statistical_Detectability.pdf>`__
   * - :abbr:`S-UNIGARD (Spatial universal Gabor residual distortion)`
     - Spatial
     -
     - —
     - `link <https://doi.org/10.1109/WIFS.2014.7084302>`__
   * - :abbr:`S-UNIWARD (Spatial universal wavelet relative distortion)`
     - Spatial
     -
     - :verified:`binghamton test/test_suniward.py::TestSUNIWARD::test_stego_reference`
     - `link <https://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf>`__
   * - :abbr:`WOW (Wavelet obtained weights)`
     - Spatial
     -
     - :verified:`binghamton test/test_wow.py::TestWOW::test_stego_reference`
     - `link <https://dde.binghamton.edu/vholub/pdf/WIFS12_Designing_Steganographic_Distortion_Using_Directional_Filters.pdf>`__
   * - :abbr:`LSB (Least significant bit replacement and matching)`
     - JPEG, spatial
     -
     - —
     -

.. rst-class:: table-footnote new-tab

Verified against reference implementations from
:raw-html:`<span id="ref-binghamton">Binghamton:</span>` `Digital Data Embedding Lab, Binghamton University <https://dde.binghamton.edu/download/stego_algorithms/>`__ (Matlab, C++);
:raw-html:`<span id="ref-troyes">Troyes:</span>` Rémi Cogranne, Troyes University of Technology (Matlab, Python, e.g., `EBS <https://codeocean.com/capsule/8552917>`__);
:raw-html:`<span id="ref-barcelona">Barcelona:</span>` Daniel Lerch Hostalot, Universitat Oberta de Catalunya (Python, `stegolab <https://github.com/daniellerch/stegolab>`__);
:raw-html:`<span id="ref-lille">Lille:</span>` Patrick Bas, CNRS, University of Lille (Python).
Click a badge to see its test.

.. container:: method-examples

   .. container:: method-example
      :name: example-ebs

      .. code-block:: python

         # simulate EBS at 0.4 bits per nonzero AC coefficient
         jpeg.Y = cl.ebs.simulate_single_channel(
             y0=jpeg.Y,      # DCT of cover luminance
             qt=jpeg.qt[0],  # quantization table
             alpha=0.4,      # embedding rate
             seed=12345,     # seed for PRNG
         )


   .. container:: method-example
      :name: example-f5

      .. code-block:: python

         # simulate F5 at 0.4 bits per nonzero AC coefficient
         jpeg.Y = cl.F5.simulate_single_channel(
             y0=jpeg.Y,   # DCT of cover luminance
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )


   .. container:: method-example
      :name: example-j-mipod

      .. code-block:: python

         # simulate J-MiPOD at 0.4 bits per nonzero AC coefficient
         jpeg.Y = cl.jmipod.simulate_single_channel(
             y0=jpeg.Y,      # DCT of cover luminance
             qt=jpeg.qt[0],  # quantization table
             alpha=0.4,      # embedding rate
             seed=12345,     # seed for PRNG
         )


   .. container:: method-example
      :name: example-j-uniward

      .. code-block:: python

         # simulate J-UNIWARD at 0.4 bits per nonzero AC coefficient
         jpeg.Y = cl.juniward.simulate_single_channel(
             y0=jpeg.Y,               # DCT of cover luminance
             qt=jpeg.qt[0],           # quantization table
             x0=im0.spatial[..., 0],  # decompressed cover luminance
             alpha=0.4,               # embedding rate
             seed=12345,              # seed for PRNG
         )


   .. container:: method-example
      :name: example-nsf5

      .. code-block:: python

         # simulate nsF5 at 0.4 bits per nonzero AC coefficient
         jpeg.Y = cl.nsF5.simulate_single_channel(
             y0=jpeg.Y,   # DCT of cover luminance
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )


   .. container:: method-example
      :name: example-uerd

      .. code-block:: python

         # simulate UERD at 0.4 bits per nonzero AC coefficient
         jpeg.Y = cl.uerd.simulate_single_channel(
             y0=jpeg.Y,      # DCT of cover luminance
             qt=jpeg.qt[0],  # quantization table
             alpha=0.4,      # embedding rate
             seed=12345,     # seed for PRNG
         )


   .. container:: method-example
      :name: example-hill

      .. code-block:: python

         # simulate HILL at 0.4 bits per pixel
         x1 = cl.hill.simulate_single_channel(
             x0=x0,       # cover pixels
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )


   .. container:: method-example
      :name: example-hugo

      .. code-block:: python

         # simulate HUGO at 0.4 bits per pixel
         x1 = cl.hugo.simulate_single_channel(
             x0=x0,       # cover pixels
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )


   .. container:: method-example
      :name: example-mipod

      .. code-block:: python

         # simulate MiPOD at 0.4 bits per pixel
         x1 = cl.mipod.simulate_single_channel(
             x0=x0,       # cover pixels
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )


   .. container:: method-example
      :name: example-s-unigard

      .. code-block:: python

         # simulate S-UNIGARD at 0.4 bits per pixel
         x1 = cl.sunigard.simulate_single_channel(
             x0=x0,       # cover pixels
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )


   .. container:: method-example
      :name: example-s-uniward

      .. code-block:: python

         # simulate S-UNIWARD at 0.4 bits per pixel
         x1 = cl.suniward.simulate_single_channel(
             x0=x0,       # cover pixels
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )


   .. container:: method-example
      :name: example-wow

      .. code-block:: python

         # simulate WOW at 0.4 bits per pixel
         x1 = cl.wow.simulate_single_channel(
             x0=x0,       # cover pixels
             alpha=0.4,   # embedding rate
             seed=12345,  # seed for PRNG
         )


   .. container:: method-example
      :name: example-lsb

      .. code-block:: python

         # simulate LSB matching at 0.4 bits per pixel
         x1 = cl.lsb.simulate(
             cover=x0,                # cover pixels
             alpha=0.4,               # embedding rate
             modify=cl.LSB_MATCHING,  # or cl.LSB_REPLACEMENT
             seed=12345,              # seed for PRNG
         )


Related projects
----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: jpeglib
      :link: https://jpeglib.readthedocs.io/

      Read and write the DCT coefficients of JPEG images,
      used by conseal for JPEG steganography.

   .. grid-item-card:: sealwatch
      :link: https://sealwatch.readthedocs.io/

      Steganalysis features and detectors,
      to detect the embedding simulated by conseal.


.. toctree::
   :hidden:
   :maxdepth: 2

   guide/index
   reference
   glossary
   faq
