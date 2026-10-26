use pyo3::prelude::*;
use numpy::{PyArray4, PyReadonlyArray2};
use numpy::ndarray::Array4;

use crate::wavelet::{correlate_separable, crop, daubechies8, pad_symmetric};


/// 1D DCT-II basis vectors (orthonormal), basis[k][n] = idct(delta_k)[n].
fn dct_basis() -> [[f64; 8]; 8] {
    let mut basis = [[0f64; 8]; 8];
    for k in 0..8 {
        let s = if k == 0 { (1.0f64 / 8.0).sqrt() } else { 0.5 };
        for n in 0..8 {
            basis[k][n] = s * (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 16.0).cos();
        }
    }
    basis
}

/// Absolute impact of a DCT basis vector on wavelet coefficients, of shape [8, 8 + len(f) - 1].
///
/// Full 1D correlation of each basis vector with the filter f.
/// The 2D impact of DCT mode (i, j) is the outer product of rows i and j.
fn wavelet_impact(basis: &[[f64; 8]; 8], f: &[f64]) -> Vec<Vec<f64>> {
    let kf = f.len();
    basis
        .iter()
        .map(|b| {
            (0..8 + kf - 1)
                .map(|t| {
                    (0..8)
                        .filter_map(|l| {
                            let m = l as isize - t as isize + kf as isize - 1;
                            (0..kf as isize).contains(&m).then(|| b[l] * f[m as usize])
                        })
                        .sum::<f64>()
                        .abs()
                })
                .collect()
        })
        .collect()
}

// Computes J-UNIWARD cost.
//
// Parameters
// ----------
// x0 : np.ndarray
//     decompressed (pixel) image (uint8) of shape [height, width], multiples of 8
// qt : np.ndarray
//     quantization table (uint16) of shape [8, 8]
// sigma : float
//     stabilizing constant
//
// Implements J-UNIWARD with the off-by-one fix (JUNIWARD_FIX_OFF_BY_ONE).
//
// Returns
// -------
// np.ndarray
//     cost for +-1 change of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
#[pyfunction]
#[pyo3(signature = (x0, qt, sigma = 0.015625))]
fn compute_cost<'py>(
    py: Python<'py>,
    x0: PyReadonlyArray2<'py, u8>,
    qt: PyReadonlyArray2<'py, u16>,
    sigma: f64,
) -> PyResult<Py<PyArray4<f64>>> {

    let x0 = x0.as_array();
    let qt = qt.as_array();
    let (h, w) = x0.dim();
    let (nby, nbx) = (h / 8, w / 8);
    let input: Vec<f64> = x0.iter().map(|&v| v as f64).collect();

    // wavelet coefficients of the padded cover, window of [h + 15, w + 15],
    // starting at 1 in 'valid' coordinates (off-by-one fix)
    let (x_pad, hp, wp) = pad_symmetric(&input, h, w, 16);
    let (hc, wc) = (h + 15, w + 15);
    let basis = dct_basis();

    let mut cost = vec![0f64; nby * nbx * 64];
    let mut tmp = [[0f64; 8]; 23];
    for (a, b) in daubechies8().iter() {
        let (rc, _, wr) = correlate_separable(&x_pad, hp, wp, a, b);
        let rc = crop(&rc, wr, 1, 1, hc, wc);
        let rc_inv: Vec<f64> = rc.iter().map(|v| 1.0 / (v.abs() + sigma)).collect();
        let p = wavelet_impact(&basis, a);  // vertical, [8, 23]
        let q = wavelet_impact(&basis, b);  // horizontal, [8, 23]

        // per block: cost += P S Q^T, S the [23, 23] window
        for m in 0..nby {
            for n in 0..nbx {
                // tmp = S Q^T, [23, 8]
                for k in 0..23 {
                    let row = &rc_inv[(8 * m + k) * wc + 8 * n..(8 * m + k) * wc + 8 * n + 23];
                    for j in 0..8 {
                        tmp[k][j] = row.iter().zip(&q[j]).map(|(s, c)| s * c).sum();
                    }
                }
                // cost += P tmp, [8, 8]
                let block = &mut cost[(m * nbx + n) * 64..(m * nbx + n + 1) * 64];
                for i in 0..8 {
                    for j in 0..8 {
                        block[i * 8 + j] += (0..23).map(|k| p[i][k] * tmp[k][j]).sum::<f64>();
                    }
                }
            }
        }
    }

    // scale by the quantization step
    for block in cost.chunks_mut(64) {
        for i in 0..8 {
            for j in 0..8 {
                block[i * 8 + j] *= qt[[i, j]] as f64;
            }
        }
    }
    let cost = Array4::from_shape_vec((nby, nbx, 8, 8), cost).unwrap();
    Ok(PyArray4::from_owned_array(py, cost).into())
}

#[pymodule]
pub fn init_juniward_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cost, m)?)?;
    Ok(())
}
