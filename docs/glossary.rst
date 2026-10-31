Glossary
========

Short definitions of the terms used in the documentation.
For the concepts behind them, see the :doc:`guide/index`,
and for JPEG specifically, the `glossary of jpeglib <https://jpeglib.readthedocs.io/en/latest/glossary.html>`__.

.. glossary::
   :sorted:

   adaptive steganography
      Steganography that prefers changing elements in textured or noisy areas,
      where changes are hard to detect.
      The preference is expressed by the :term:`cost` of each element.

   bpc
      Bits per coefficient, an :term:`embedding rate` relative to all DCT coefficients.

   bpe
      Bits per element, an :term:`embedding rate` relative to all pixel values, over all color channels.

   bpnzAC
      Bits per nonzero AC coefficient, the usual :term:`embedding rate` of JPEG steganography.

   bpp
      Bits per pixel, the usual :term:`embedding rate` of spatial steganography.

   change probability
      The probability that an element is changed by the embedding, e.g., by +1 or -1.
      The :term:`simulator` computes it from the :term:`cost`.

   coding loss
      How much a practical code falls short of :term:`optimal coding`,
      i.e., how many more changes it needs for the same :term:`payload`.

   cost
      The detection risk of changing a given element, denoted ρ.
      The methods differ mainly in how they define the cost.

   distortion
      The total sum of the :term:`costs <cost>` of the changes, for additive costs.
      The embedding minimizes it, or its expectation in the :term:`simulator`.

   cover
      The image before the embedding.

   DCT coefficient
      A coefficient of the discrete cosine transform of an 8x8 block of a JPEG image,
      quantized by the :term:`quantization table`.
      The first coefficient of each block is the DC, the other 63 are AC.

   distortion-limited sender
      A sender that embeds as large a :term:`payload` as possible,
      within a given expected :term:`distortion`.

   embedding efficiency
      The number of message bits embedded per change, denoted e.

   embedding rate
      The :term:`payload` relative to the size of the :term:`cover`, denoted α,
      e.g., in :term:`bpp` or :term:`bpnzAC`.

   lambda
      The parameter λ of the :term:`change probability` p ~ exp(-λρ),
      chosen by the :term:`simulator` to meet the :term:`payload`, or the :term:`distortion`.

   optimal coding
      Coding at the theoretical bound, with the highest possible :term:`embedding efficiency`.
      The :term:`simulator` assumes it by default.

   payload
      The length of the message, in bits.

   payload-limited sender
      A sender that embeds a given :term:`payload` with the least expected :term:`distortion`.
      The default sender of the :term:`simulator`.

   quantization table
      The 8x8 table of step sizes, by which a JPEG image divides its DCT coefficients.

   reference implementation
      The implementation of a method by its authors.
      conseal is tested to produce the same costs or stego images.

   selection channel
      The :term:`change probabilities <change probability>` of the embedding.
      Steganalysis that knows them detects better, which is called selection-channel-aware.

   side-informed embedding
      Embedding that uses a precover, e.g., the uncompressed image before JPEG compression,
      to lower the cost of elements with a large rounding error.

   simulator
      Simulates the changes of the embedding without embedding a message,
      as :term:`optimal coding` would make them.

   steganalysis
      The detection of steganography.

   stego
      The image after the embedding.

   syndrome-trellis codes
   STC
      Practical codes, which embed close to :term:`optimal coding`,
      with costs of each element.

   wet element
      An element that must not change, e.g., the DC coefficient, or a saturated pixel.
      It gets the :term:`wet cost`.

   wet cost
      A large cost given to :term:`wet elements <wet element>`, e.g., 10\ :sup:`10`,
      so that they practically never change.

   wet-paper codes
      Codes that embed without changing :term:`wet elements <wet element>`,
      while the receiver does not need to know which elements were wet.
