[![PyPI version](https://badge.fury.io/py/conseal.svg)](https://pypi.org/project/conseal/)
[![Commit CI/CD](https://github.com/uibk-uncover/conseal/actions/workflows/on_commit.yml/badge.svg?branch=dev)](https://github.com/uibk-uncover/conseal/actions/workflows/on_commit.yml)
[![Release CI/CD](https://github.com/uibk-uncover/conseal/actions/workflows/on_release.yml/badge.svg)](https://github.com/uibk-uncover/conseal/actions/workflows/on_release.yml)
[![Documentation Status](https://readthedocs.org/projects/conseal/badge/?version=latest)](https://conseal.readthedocs.io/)
[![PyPI downloads](https://img.shields.io/pypi/dm/conseal)](https://pypi.org/project/conseal/)
[![Stars](https://img.shields.io/github/stars/uibk-uncover/conseal.svg)](https://GitHub.com/uibk-uncover/conseal)
[![Contributors](https://img.shields.io/github/contributors/uibk-uncover/conseal)](https://GitHub.com/uibk-uncover/conseal)
[![Wheel](https://img.shields.io/pypi/wheel/conseal)](https://pypi.org/project/conseal/)
[![Status](https://img.shields.io/pypi/status/conseal)](https://pypi.com/project/conseal/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/conseal/)
[![Last commit](https://img.shields.io/github/last-commit/uibk-uncover/conseal)](https://GitHub.com/uibk-uncover/conseal)

<img src="https://raw.githubusercontent.com/uibk-uncover/conseal/main/docs/static/seal.png" width="300" />

# conseal

Python package, containing implementations of modern image steganographic algorithms.

> :warning: The package only simulates the embedding, which is useful for steganalysis research. We do not provide any end-to-end steganography method.


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

## Contents

| Steganography method | Domain | Reference |
| --- | --- | --- |
| nsF5: no-shrinkage F5 | JPEG | [Original F5 algorithm](https://doi.org/10.1007/3-540-45496-9_21), [no-shrinkage extension](https://doi.org/10.1145/1288869.1288872) |
| EBS: entropy block steganography | JPEG | [Reference](https://doi.org/10.1109/ICASSP.2012.6288246) |
| UERD: uniform embedding revisited distortion | JPEG | [Reference](https://doi.org/10.1109/TIFS.2015.2473815) |
| J-UNIWARD: JPEG-domain universal wavelet relative distortion | JPEG | [Reference](https://doi.org/10.1186/1687-417X-2014-1) |

## Usage

Import the library in Python 3

```python
import conseal as cl
```

This package currently contains the three JPEG steganography methods J-UNIWARD, UERD, and nsF5. The following examples show how to embed a JPEG cover image `cover.jpeg` with an embedding rate of 0.4 bits per non-zero AC coefficient (bpnzAC):

- J-UNIWARD

```python
# load cover
im_spatial = jpeglib.read_spatial("cover.jpeg", jpeglib.JCS_GRAYSCALE)
im_dct = jpeglib.read_dct("cover.jpeg")

# embed J-UNIWARD 0.4
im_dct.Y = cl.juniward.simulate_single_channel(
    cover_spatial=im_spatial.spatial[..., 0],
    cover_dct_coeffs=im_dct.Y,
    quantization_table=im_dct.qt[0],
    embedding_rate=0.4,
    seed=12345
)

# save result as stego image
im_dct.write_dct("stego.jpeg")
```

- UERD

```python
# load cover
im_dct = jpeglib.read_dct("cover.jpeg")

# embed UERD 0.4
im_dct.Y = cl.uerd.simulate_single_channel(
    cover_dct_coeffs=im_dct.Y,
    quantization_table=im_dct.qt[0],
    embedding_rate=0.4,
    seed=12345
)

# save result as stego image
im_dct.write_dct("stego.jpeg")
```

- nsF5

```python
# load cover
im_dct = jpeglib.read_dct("cover.jpeg")

# embed nsF5 0.4 bpnzAC
im_dct.Y = cl.nsF5.simulate_single_channel(
    cover_dct_coeffs=im_dct.Y,
    embedding_rate=0.4,
    seed=12345
)

# save result as stego image
im_dct.write_dct("stego.jpeg")
```

- EBS

```python
# load cover
im_dct = jpeglib.read_dct("cover.jpeg")

# embed EBS 0.4 bpnzAC
im_dct.Y = cl.ebs.simulate_single_channel(
    cover_dct_coeffs=im_dct.Y,
    quantization_table=im_dct.qt[0],
    embedding_rate=0.4,
    seed=12345
)

# save result as stego image
im_dct.write_dct("stego.jpeg")
```

- HUGO

```python
# load cover
cover_spatial = np.array(Image.open("cover.png"))

# embed HUGO 0.4 bpnzAC
stego_spatial = cl.hugo.simulate_single_channel(
    cover_spatial=cover_spatial,
    embedding_rate=0.4,
    seed=12345)

# save result as stego image
Image.fromarray(stego_spatial).save("stego.png")
```

- HILL

```python
# load cover
cover_spatial = np.array(Image.open("cover.png"))

# embed HUGO 0.4 bpnzAC
stego_spatial = cl.hill.simulate_single_channel(
    cover_spatial=cover_spatial,
    embedding_rate=0.4,
    seed=12345)

# save result as stego image
Image.fromarray(stego_spatial).save("stego.png")
```

## Acknowledgements and Disclaimer

Developed by [Martin Benes](https://github.com/martinbenes1996) and [Benedikt Lorch](https://github.com/btlorch/), University of Innsbruck, 2023.

The J-UNIWARD and nsF5 implementations in this package are based on the original Matlab code provided by the Digital Data Embedding Lab at Binghamton University.
We also thank Patrick Bas and Rémi Cogranne for sharing their implementations of UERD and EBS with us.

We have made our best effort to ensure that our implementations produce identical results as the original Matlab implementations. However, it is the user's responsibility to verify this.