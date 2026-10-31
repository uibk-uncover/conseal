Generating a dataset
====================

A typical experiment embeds a folder of covers at a fixed payload,
and trains a steganalysis detector on the cover-stego pairs,
e.g., with our sister project `sealwatch <https://sealwatch.readthedocs.io/>`__.

Tips:

- Use a seed per image, see `Seed per image`_.
- For JPEG, write the covers with jpeglib too, see `JPEG dataset`_.
- Set the payload in the unit of the domain, e.g., bpnzAC for JPEG, see :doc:`payload`.
- Save spatial stego in a lossless format, such as PNG.
- Keep the file names of cover and stego the same, to pair them.
- Embed the images in parallel, the methods are single-threaded, see `Parallel processing`_.
- To compare with published results, use the original implementations, see :doc:`reproducibility`.


Spatial dataset
---------------

.. code-block:: python

   from pathlib import Path
   import conseal as cl
   import numpy as np
   from PIL import Image

   cover_dir, stego_dir = Path("cover"), Path("stego_hill_0.4")
   stego_dir.mkdir(exist_ok=True)

   for cover_path in sorted(cover_dir.glob("*.png")):
       x0 = np.array(Image.open(cover_path))
       x1 = cl.hill.simulate_single_channel(
           x0=x0,
           alpha=0.4,
           seed=cl.tools.password_to_seed(cover_path.name),  # seed per image
       )
       Image.fromarray(x1).save(stego_dir / cover_path.name)


Seed per image
--------------

Give each image its own seed.
With the same seed, all images draw the same random numbers,
so images of the same size change in correlated positions,
which a detector may learn.

``cl.tools.password_to_seed`` derives the seed from the file name,
so the stego of an image is the same,
regardless of the order or the subset of the processed images.
Its seeds are large integers;
for ``generator='MT19937'``, take them modulo ``2**32``.


JPEG dataset
------------

For JPEG, write the cover with jpeglib too,
so that cover and stego differ only by the embedding,
see the warning in :doc:`loading`.

.. code-block:: python

   import jpeglib

   cover_dir = Path("cover_jpeg")
   cover_out, stego_out = Path("cover_jpeglib"), Path("stego_uerd_0.4")
   cover_out.mkdir(exist_ok=True)
   stego_out.mkdir(exist_ok=True)

   for cover_path in sorted(cover_dir.glob("*.jpg")):
       jpeg = jpeglib.read_dct(cover_path)
       jpeg.write_dct(cover_out / cover_path.name)  # cover, as jpeglib writes it

       jpeg.Y = cl.uerd.simulate_single_channel(
           y0=jpeg.Y,
           qt=jpeg.qt[0],
           alpha=0.4,
           seed=cl.tools.password_to_seed(cover_path.name),
       )
       jpeg.write_dct(stego_out / cover_path.name)


Parallel processing
-------------------

The methods run on a single thread.
To use more cores, embed several images in parallel,
e.g., with the standard library.
Thanks to the seed per image, the result does not depend on the scheduling.

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor

   def embed(cover_path):
       x0 = np.array(Image.open(cover_path))
       x1 = cl.hill.simulate_single_channel(
           x0=x0,
           alpha=0.4,
           seed=cl.tools.password_to_seed(cover_path.name),
       )
       Image.fromarray(x1).save(stego_dir / cover_path.name)

   if __name__ == "__main__":
       with ProcessPoolExecutor() as pool:
           list(pool.map(embed, sorted(cover_dir.glob("*.png"))))
