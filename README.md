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
| F5 | JPEG | [Reference](https://doi.org/10.1007/3-540-45496-9_21) |
| nsF5: no-shrinkage F5 | JPEG | [Reference](https://dde.binghamton.edu/kodovsky/pdf/Fri07-ACM.pdf) |
| EBS: entropy block steganography | JPEG | [Reference](https://doi.org/10.1109/ICASSP.2012.6288246) |
| UERD: uniform embedding revisited distortion | JPEG | [Reference](https://doi.org/10.1109/TIFS.2015.2473815) |
| J-UNIWARD: JPEG-domain universal wavelet relative distortion | JPEG | [Reference](https://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf) |
| LSB: least significant bit | Spatial / JPEG | |
| HILL: high-low-low | Spatial | [Reference](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569891955.pdf) |
| HUGO: highly undetectable stego | Spatial | [Reference](http://agents.fel.cvut.cz/stegodata/pdfs/Pev10-Hugo.pdf) |
| MiPOD: minimizing the power of optimal detector | Spatial | [Reference](https://dde.binghamton.edu/vsedighi/pdf/TIFS2015_Content_Adaptive_Steganography_by_Minimizing_Statistical_Detectability.pdf) |
| S-UNIWARD: spatial-domain universal wavelet relative distortion | spatial | [Reference](https://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf) |
| WOW: wavelet obtained weights | spatial | [Reference](https://dde.binghamton.edu/vholub/pdf/WIFS12_Designing_Steganographic_Distortion_Using_Directional_Filters.pdf) |

## Usage

Import the library in Python 3

```python
import conseal as cl
```

This package currently contains the three JPEG steganography methods J-UNIWARD, UERD, and nsF5. The following examples show how to embed a JPEG cover image `cover.jpeg` with an embedding rate of 0.4 bits per non-zero AC coefficient (bpnzAC):



- F5

```python
# load cover
jpeg = jpeglib.read_dct("cover.jpeg")

# embed F5 0.4 bpnzAC
jpeg.Y = cl.F5.simulate_single_channel(
    y0=jpeg.Y,
    alpha=0.4,
    seed=12345)

# save result as stego image
jpeg.write_dct("stego.jpeg")
```

- nsF5

```python
# load cover
jpeg = jpeglib.read_dct("cover.jpeg")

# embed nsF5 0.4 bpnzAC
jpeg.Y = cl.nsF5.simulate_single_channel(
    y0=jpeg.Y,
    alpha=0.4,
    seed=12345)

# save result as stego image
jpeg.write_dct("stego.jpeg")
```

- EBS

```python
# load cover
jpeg = jpeglib.read_dct("cover.jpeg")

# embed EBS 0.4 bpnzAC
jpeg.Y = cl.ebs.simulate_single_channel(
    y0=jpeg.Y,
    qt=jpeg.qt[0],
    alpha=0.4,
    seed=12345)

# save result as stego image
jpeg.write_dct("stego.jpeg")
```

- UERD

```python
# load cover
jpeg = jpeglib.read_dct("cover.jpeg")

# embed UERD 0.4
jpeg.Y = cl.uerd.simulate_single_channel(
    y0=jpeg.Y,
    qt=jpeg.qt[0],
    embedding_rate=0.4,
    seed=12345)

# save result as stego image
jpeg.write_dct("stego.jpeg")
```

- J-UNIWARD

```python
# load cover
im0 = jpeglib.read_spatial("cover.jpeg", jpeglib.JCS_GRAYSCALE)
jpeg = jpeglib.read_dct("cover.jpeg")

# embed J-UNIWARD 0.4
jpeg.Y = cl.juniward.simulate_single_channel(
    x0=im0.spatial[..., 0],
    y0=jpeg.Y,
    qt=jpeg.qt[0],
    alpha=0.4,
    seed=12345)

# save result as stego image
jpeg.write_dct("stego.jpeg")
```

- HUGO

```python
# load cover
x0 = np.array(Image.open("cover.png"))

# embed HUGO 0.4 bpnzAC
x1 = cl.hugo.simulate_single_channel(
    x0=x0,
    alpha=0.4,
    seed=12345)

# save result as stego image
Image.fromarray(x1).save("stego.png")
```

- WOW


```python
# load cover
x0 = np.array(Image.open("cover.png"))

# embed WOW 0.4 bpnzAC
x1 = cl.wow.simulate_single_channel(
    x0=x0,
    alpha=0.4,
    seed=12345)

# save result as stego image
Image.fromarray(x1).save("stego.png")
```

- S-UNIWARD

```python
# load cover
x0 = np.array(Image.open("cover.png"))

# embed S-UNIWARD 0.4 bpnzAC
x1 = cl.suniward.simulate_single_channel(
    x0=x0,
    alpha=0.4,
    seed=12345)

# save result as stego image
Image.fromarray(x1).save("stego.png")
```

- MiPOD

```python
# load cover
x0 = np.array(Image.open("cover.png"))

# embed MiPOD 0.4 bpnzAC
x1 = cl.mipod.simulate_single_channel(
    x0=x0,
    alpha=0.4,
    seed=12345)

# save result as stego image
Image.fromarray(x1).save("stego.png")
```

- HILL

```python
# load cover
x0 = np.array(Image.open("cover.png"))

# embed HUGO 0.4 bpnzAC
x1 = cl.hill.simulate_single_channel(
    x0=x0,
    alpha=0.4,
    seed=12345)

# save result as stego image
Image.fromarray(x1).save("stego.png")
```

## Acknowledgements and Disclaimer

Developed by [Martin Benes](https://github.com/martinbenes1996) and [Benedikt Lorch](https://github.com/btlorch/), University of Innsbruck, 2023.

The J-UNIWARD and nsF5 implementations in this package are based on the original Matlab code provided by the Digital Data Embedding Lab at Binghamton University.
We also thank Patrick Bas and Rémi Cogranne for sharing their implementations of UERD and EBS with us.

We have made our best effort to ensure that our implementations produce identical results as the original Matlab implementations. However, it is the user's responsibility to verify this.