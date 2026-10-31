[![PyPI version](https://badge.fury.io/py/conseal.svg)](https://pypi.org/project/conseal/)
[![Commit CI/CD](https://github.com/uibk-uncover/conseal/actions/workflows/on_commit.yml/badge.svg?branch=main)](https://github.com/uibk-uncover/conseal/actions/workflows/on_commit.yml)
[![Documentation Status](https://readthedocs.org/projects/conseal/badge/?version=latest)](https://conseal.readthedocs.io/)
[![PyPI downloads](https://img.shields.io/pypi/dm/conseal)](https://pypi.org/project/conseal/)
[![License](https://img.shields.io/github/license/uibk-uncover/conseal)](https://github.com/uibk-uncover/conseal/blob/main/LICENSE)

<img src="https://raw.githubusercontent.com/uibk-uncover/conseal/main/docs/static/seal.png" width="200" />

# conseal

Making research in steganography and steganalysis reproducible and accessible.

**conseal** is a Python package implementing the cost functions and embedding simulators
of modern steganographic methods, for JPEG and spatial images.

- **13 methods**, from LSB to J-UNIWARD and MiPOD,
  most verified against the reference implementations of their authors, down to the stego image.
- **Three levels of API**: one call from cover to stego, or separate costs, probabilities and sampling.
- **Reproducible**: seeded simulation, with the original implementations and random number generators of the references.
- **Fast**: Rust backends for the costs of HILL, J-UNIWARD, S-UNIWARD, UERD and WOW.

Documentation: https://conseal.readthedocs.io


## Installation

```bash
pip install conseal
```

Prebuilt wheels contain the Rust backend.
Building from source, e.g., on other platforms or from a clone of the repository,
requires the [Rust toolchain](https://rustup.rs/).


## Quick start

JPEG steganography, with [jpeglib](https://jpeglib.readthedocs.io/) to access the DCT coefficients:

```python
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
```

Spatial steganography:

```python
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
```

An example for each method is in the [documentation](https://conseal.readthedocs.io/en/latest/#methods).


## Methods

| Method | Domain | Paper |
| --- | --- | --- |
| EBS: entropy block steganography | JPEG | [link](https://doi.org/10.1109/ICASSP.2012.6288246) |
| F5 | JPEG | [link](https://doi.org/10.1007/3-540-45496-9_21) |
| J-MiPOD: JPEG minimizing the power of optimal detector | JPEG | [link](https://doi.org/10.1145/3369412.3395075) |
| J-UNIWARD: JPEG universal wavelet relative distortion | JPEG | [link](https://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf) |
| nsF5: no-shrinkage F5 | JPEG | [link](https://dde.binghamton.edu/kodovsky/pdf/Fri07-ACM.pdf) |
| UERD: uniform embedding revisited distortion | JPEG | [link](https://doi.org/10.1109/TIFS.2015.2473815) |
| HILL: high-low-low | spatial | [link](https://doi.org/10.1109/ICIP.2014.7025854) |
| HUGO: highly undetectable stego | spatial | [link](https://doi.org/10.1007/978-3-642-16435-4_13) |
| MiPOD: minimizing the power of optimal detector | spatial | [link](https://dde.binghamton.edu/vsedighi/pdf/TIFS2015_Content_Adaptive_Steganography_by_Minimizing_Statistical_Detectability.pdf) |
| S-UNIGARD: spatial universal Gabor residual distortion | spatial | [link](https://doi.org/10.1109/WIFS.2014.7084302) |
| S-UNIWARD: spatial universal wavelet relative distortion | spatial | [link](https://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf) |
| WOW: wavelet obtained weights | spatial | [link](https://dde.binghamton.edu/vholub/pdf/WIFS12_Designing_Steganographic_Distortion_Using_Directional_Filters.pdf) |
| LSB: least significant bit replacement and matching | JPEG, spatial | |

Which methods are verified against which reference implementation,
with links to the unit tests, is shown on the [title page of the documentation](https://conseal.readthedocs.io/en/latest/#methods).


## Scope

conseal simulates the embedding changes, which is what steganalysis research needs.
For nsF5, it also embeds and extracts real messages with wet-paper codes.
It is not a stego system for real-world use:
it has no encryption, no message framing, and no key management.


## Contributing

Contributions are welcome, see [CONTRIBUTING.md](https://github.com/uibk-uncover/conseal/blob/main/CONTRIBUTING.md).
Please report bugs and deviations from reference implementations as [GitHub issues](https://github.com/uibk-uncover/conseal/issues).


## Acknowledgements

Developed by [Martin Beneš](https://github.com/martinbenes1996) and [Benedikt Lorch](https://github.com/btlorch/), University of Innsbruck.

The J-UNIWARD and nsF5 implementations are based on the original Matlab code by the Digital Data Embedding Lab at Binghamton University.
We also thank Patrick Bas and Rémi Cogranne for sharing their implementations of UERD and EBS with us.

conseal is released under the [Mozilla Public License 2.0](https://github.com/uibk-uncover/conseal/blob/main/LICENSE).
