use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use numpy::ndarray::Array2;

use crate::wavelet::{convolve_separable, daubechies8, pad_symmetric};


/// WOW directional residual xi for one filter pair (a, b), of shape [h, w].
///
/// xi = |x * K| * |rot180(K)|, with K = a b^T
fn residual(x_pad: &[f64], hp: usize, wp: usize, h: usize, w: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    // residual
    let (r, hr, wr) = convolve_separable(x_pad, hp, wp, a, b);
    let r_abs: Vec<f64> = r.iter().map(|v| v.abs()).collect();
    // rotate 180 + absolute kernel, separable as well
    let a_rot: Vec<f64> = a.iter().rev().map(|v| v.abs()).collect();
    let b_rot: Vec<f64> = b.iter().rev().map(|v| v.abs()).collect();
    let (xi, _, wx) = convolve_separable(&r_abs, hr, wr, &a_rot, &b_rot);
    // crop, matches mode='same' on the padded image with offset 1
    let mut out = Vec::with_capacity(h * w);
    for i in 0..h {
        out.extend_from_slice(&xi[(i + 1) * wx + 1..(i + 1) * wx + 1 + w]);
    }
    out
}

// Computes WOW cost.
//
// Parameters
// ----------
// x0 : np.ndarray
//     uncompressed (pixel) cover image of shape [height, width]
// p : float
//     power of the aggregation
//
// Returns
// -------
// np.ndarray
//     cost for +-1 change of shape [height, width]
#[pyfunction]
#[pyo3(signature = (x0, p = -1.0))]
fn compute_cost<'py>(py: Python<'py>, x0: PyReadonlyArray2<'py, u8>, p: f64)
    -> PyResult<Py<PyArray2<f64>>> {

    let x0 = x0.as_array();
    let (h, w) = x0.dim();
    let input: Vec<f64> = x0.iter().map(|&v| v as f64).collect();

    // pad once, both convolutions are 'valid'
    let (x_pad, hp, wp) = pad_symmetric(&input, h, w, 16);

    // directional residuals
    let xi: Vec<Vec<f64>> = daubechies8()
        .iter()
        .map(|(a, b)| residual(&x_pad, hp, wp, h, w, a, b))
        .collect();

    // aggregate: rho = (sum_i xi_i^p)^(-1/p)
    let rho: Vec<f64> = (0..h * w)
        .map(|k| {
            xi.iter()
                .map(|x| x[k].max(f64::EPSILON).powf(p))
                .sum::<f64>()
                .powf(-1.0f64 / p)
        })
        .collect();
    let rho = Array2::from_shape_vec((h, w), rho).unwrap();
    Ok(PyArray2::from_owned_array(py, rho).into())
}

#[pymodule]
pub fn init_wow_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cost, m)?)?;
    Ok(())
}
