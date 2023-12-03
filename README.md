[![PyPI version](https://badge.fury.io/py/conseal.svg)](https://pypi.org/project/conseal/)
[![Commit CI/CD](https://github.com/uibk-uncover/conseal/actions/workflows/on_commit.yml/badge.svg?branch=master)](https://github.com/uibk-uncover/conseal/actions/workflows/on_commit.yml)
[![Release CI/CD](https://github.com/uibk-uncover/conseal/actions/workflows/on_release.yml/badge.svg)](https://github.com/uibk-uncover/conseal/actions/workflows/on_release.yml)
[![Documentation Status](https://readthedocs.org/projects/conseal/badge/?version=latest)](https://conseal.readthedocs.io/)
[![PyPI downloads](https://img.shields.io/pypi/dm/conseal)](https://pypi.org/project/conseal/)
[![Stars](https://img.shields.io/github/stars/uibk-uncover/conseal.svg)](https://GitHub.com/uibk-uncover/conseal)
[![Contributors](https://img.shields.io/github/contributors/uibk-uncover/conseal)](https://GitHub.com/uibk-uncover/conseal)
[![Wheel](https://img.shields.io/pypi/wheel/conseal)](https://pypi.org/project/conseal/)
[![Status](https://img.shields.io/pypi/status/conseal)](https://pypi.com/project/conseal/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/conseal/)
[![Last commit](https://img.shields.io/github/last-commit/uibk-uncover/conseal)](https://GitHub.com/uibk-uncover/conseal)

<img src="docs/seal.png" width="25%"/>

# conseal

Python package, containing implementations of modern image steganographic algorithms.

> :warning: The package can perform simulated embedding only, which is useful for steganalysis research. We will not provide end-to-end implementation of steganography.




## Installation

Simply install the package with pip3


```bash
pip3 install conseal
```

or using the cloned repository

```bash
git clone https://github.com/uibk-uncover/conseal/
cd conseal
pip3 install .
```


## Usage

Import the library in Python 3

```python
import conseal as cl
```

Simulated embedding at 0.4 bpnzAC into `"cover.jpeg"`` is implemented for following algorithms:

- J-UNIWARD

```python
# load cover
im = jpeglib.read_spatial("cover.jpeg", jpeglib.JCS_GRAYSCALE)
jpeg = jpeglib.read_dct("cover.jpeg")
# embed J-UNIWARD 0.4
jpeg.Y = cl.juniward.simulate_single_channel(
    cover_spatial=im.spatial[..., 0],  # spatial
    cover_dct_coeffs=jpeg.Y,
    quantization_table=jpeg.qt[0],
    embedding_rate=0.4,
    seed=12345
)
# save result as stego image
jpeg.write_dct("stego.jpeg")
```

- UERD

```python
# load cover
jpeg = jpeglib.read_dct("cover.jpeg")
# embed UERD 0.4
jpeg.Y = cl.uerd.simulate_single_channel(
    cover_dct_coeffs=jpeg.Y,
    quantization_table=jpeg.qt[0],
    embedding_rate=0.4,
    seed=12345
)
# save result as stego image
jpeg.write_dct("stego.jpeg")
```

- nsF5

```python
# load cover
jpeg = jpeglib.read_dct("cover.jpeg")
# embed UERD 0.4
jpeg.Y = cl.nsF5.simulate_single_channel(
    cover_dct_coeffs=jpeg.Y,
    quantization_table=jpeg.qt[0],
    embedding_rate=0.4,
    seed=12345
)
# save result as stego image
jpeg.write_dct("stego.jpeg")
```



## Credits

Developed by [Martin Benes](https://github.com/martinbenes1996) and [Benedikt Lorch](https://github.com/btlorch/), University of Innsbruck, 2023.
