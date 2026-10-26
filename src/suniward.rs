use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use numpy::ndarray::Array2;

use crate::wavelet::{convolve_separable, correlate_separable, crop, daubechies8, pad_symmetric};


/// S-UNIWARD suitability xi for one filter pair (a, b), of shape [h, w].
///
/// xi = 1 / (|x * K| + sigma) (*) |K|, with K = a b^T,
/// * convolution and (*) correlation.
fn suitability(x_pad: &[f64], hp: usize, wp: usize, h: usize, w: usize, a: &[f64], b: &[f64], sigma: f64) -> Vec<f64> {
    // residual
    let (r, hr, wr) = convolve_separable(x_pad, hp, wp, a, b);
    let r_inv: Vec<f64> = r.iter().map(|v| 1.0 / (v.abs() + sigma)).collect();
    // correlate with the absolute kernel, separable as well
    let a_abs: Vec<f64> = a.iter().map(|v| v.abs()).collect();
    let b_abs: Vec<f64> = b.iter().map(|v| v.abs()).collect();
    let (xi, _, wx) = correlate_separable(&r_inv, hr, wr, &a_abs, &b_abs);
    // crop, matches mode='same' on the padded image with offset 1
    crop(&xi, wx, 1, 1, h, w)
}

// Computes S-UNIWARD cost.
//
// Parameters
// ----------
// x0 : np.ndarray
//     uncompressed (pixel) cover image (uint8) of shape [height, width]
// sigma : float
//     stabilizing constant
//
// Returns
// -------
// np.ndarray
//     cost for +-1 change of shape [height, width]
#[pyfunction]
#[pyo3(signature = (x0, sigma = 1.0))]
fn compute_cost<'py>(py: Python<'py>, x0: PyReadonlyArray2<'py, u8>, sigma: f64)
    -> PyResult<Py<PyArray2<f64>>> {

    let x0 = x0.as_array();
    let (h, w) = x0.dim();
    let input: Vec<f64> = x0.iter().map(|&v| v as f64).collect();

    // pad once, both convolutions are 'valid'
    let (x_pad, hp, wp) = pad_symmetric(&input, h, w, 16);

    // sum of suitabilities over the three directions
    let mut rho = vec![0f64; h * w];
    for (a, b) in daubechies8().iter() {
        let xi = suitability(&x_pad, hp, wp, h, w, a, b, sigma);
        for (r, x) in rho.iter_mut().zip(&xi) {
            *r += x;
        }
    }
    let rho = Array2::from_shape_vec((h, w), rho).unwrap();
    Ok(PyArray2::from_owned_array(py, rho).into())
}

/// 2D correlation with zero fill, mode='same' as in SciPy, of shape [h, w].
///
/// out[i, j] = sum_{u, v} input[i + u - cy, j + v - cx] * f[u, v], with c = (k - 1) / 2
fn correlate_same(input: &[f64], h: usize, w: usize, f: &[f64], m: usize, n: usize) -> Vec<f64> {
    let (cy, cx) = (((m - 1) / 2) as isize, ((n - 1) / 2) as isize);
    let (hi, wi) = (h as isize, w as isize);
    let mut out = vec![0f64; h * w];
    // row by row, so the output row stays in cache,
    // accumulating tap by tap, contiguous over columns
    for i in 0..hi {
        let dst_row = &mut out[(i * wi) as usize..((i + 1) * wi) as usize];
        for u in 0..m as isize {
            let ii = i + u - cy;
            if ii < 0 || ii >= hi {
                continue;
            }
            let src_row = &input[(ii * wi) as usize..((ii + 1) * wi) as usize];
            for v in 0..n as isize {
                let c = f[(u as usize) * n + v as usize];
                let (j0, j1) = ((cx - v).max(0), (wi + cx - v).min(wi));
                if c == 0.0 || j0 >= j1 {
                    continue;
                }
                let dst = &mut dst_row[j0 as usize..j1 as usize];
                let src = &src_row[(j0 + v - cx) as usize..(j1 + v - cx) as usize];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += c * s;
                }
            }
        }
    }
    out
}

// Computes S-UNIWARD cost with arbitrary 2D filters, e.g., Gabor for S-UNIGARD.
//
// Matches the non-separable Python implementation.
//
// Parameters
// ----------
// x0 : np.ndarray
//     uncompressed (pixel) cover image (uint8) of shape [height, width]
// filters : list of np.ndarray
//     2D filters
// sigma : float
//     stabilizing constant
//
// Returns
// -------
// np.ndarray
//     cost for +-1 change of shape [height, width]
#[pyfunction]
#[pyo3(signature = (x0, filters, sigma = 1.0))]
fn compute_cost_filters<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray2<'py, u8>,
    filters: Vec<PyReadonlyArray2<'py, f64>>,
    sigma: f64,
) -> PyResult<Py<PyArray2<f64>>> {

    let x0 = x0.as_array();
    let (h, w) = x0.dim();
    let input: Vec<f64> = x0.iter().map(|&v| v as f64).collect();

    // pad by the largest filter dimension
    let pad = filters.iter().flat_map(|f| f.shape().to_vec()).max().unwrap_or(0);
    let (x_pad, hp, wp) = pad_symmetric(&input, h, w, pad);

    let mut rho = vec![0f64; h * w];
    for f in filters.iter() {
        let f = f.as_array();
        let (m, n) = f.dim();
        let k: Vec<f64> = f.iter().copied().collect();
        // residual: correlation with the flipped filter, i.e., convolution
        let k_flip: Vec<f64> = k.iter().rev().copied().collect();
        let r = correlate_same(&x_pad, hp, wp, &k_flip, m, n);
        let r_inv: Vec<f64> = r.iter().map(|v| 1.0 / (v.abs() + sigma)).collect();
        // suitability: correlation with the absolute filter
        let k_abs: Vec<f64> = k.iter().map(|v| v.abs()).collect();
        let xi = correlate_same(&r_inv, hp, wp, &k_abs, m, n);
        // shift for even filters, then remove padding
        let top = pad - (m % 2 == 0) as usize;
        let left = pad - (n % 2 == 0) as usize;
        for (r, x) in rho.iter_mut().zip(crop(&xi, wp, top, left, h, w)) {
            *r += x;
        }
    }
    let rho = Array2::from_shape_vec((h, w), rho).unwrap();
    Ok(PyArray2::from_owned_array(py, rho).into())
}

#[pymodule]
pub fn init_suniward_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cost, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cost_filters, m)?)?;
    Ok(())
}
