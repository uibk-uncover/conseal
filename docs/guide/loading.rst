Loading images
==============

conseal works on numpy arrays.
How you get them depends on the domain of the method.


Spatial images
--------------

Spatial methods, such as HILL or S-UNIWARD, embed into pixels.
Load the cover with `pillow <https://pypi.org/project/pillow/>`__,
as a grayscale array of shape [height, width].

.. code-block:: python

   import numpy as np
   from PIL import Image

   x0 = np.array(Image.open("cover.png"))

The stego is an array of the same shape.
Save it in a lossless format, such as PNG,
since lossy compression would destroy the embedding changes.

.. code-block:: python

   Image.fromarray(x1).save("stego.png")


JPEG images
-----------

JPEG methods, such as J-UNIWARD or UERD, embed into the quantized DCT coefficients.
Read them with our sister project `jpeglib <https://jpeglib.readthedocs.io/>`__,
which gives the luminance as an array of 8x8 blocks,
of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8].

.. code-block:: python

   import jpeglib

   jpeg = jpeglib.read_dct("cover.jpeg")
   y0 = jpeg.Y          # quantized DCT of luminance
   qt = jpeg.qt[0]      # its quantization table

Some methods, such as J-UNIWARD, also need the decompressed cover.

.. code-block:: python

   im0 = jpeglib.read_spatial("cover.jpeg", jpeglib.JCS_GRAYSCALE)
   x0 = im0.spatial[..., 0]

To save the stego, replace the coefficients and write them back,
without recompressing the image.

.. code-block:: python

   jpeg.Y = y1
   jpeg.write_dct("stego.jpeg")

.. warning::

   jpeglib writes the DCT coefficients exactly,
   but it does not copy the structure of the original file.
   The stego may differ from the cover beyond the embedding changes,
   e.g., in the markers, the metadata or the Huffman tables.
   Such differences reveal the stego regardless of the embedding.

   This is not a problem for most experiments,
   which compare the DCT coefficients,
   or write the cover with jpeglib too, without changes.

   .. code-block:: python

      jpeglib.read_dct("cover.jpeg").write_dct("cover_jpeglib.jpeg")

   In real-world applications, however, it remains an open problem,
   since the files written by jpeglib may differ
   from the files of cameras or common software.


DCT from jpegio
---------------

If you read the DCT with `jpegio <https://pypi.org/project/jpegio/>`__,
you get a 2D array instead of blocks.
Convert it to the 4D shape and back as follows.

.. code-block:: python

   import conseal as cl

   y0 = cl.tools.jpegio_to_jpeglib(y0_2d)
   y1_2d = cl.tools.jpeglib_to_jpegio(y1)
