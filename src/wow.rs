use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use numpy::ndarray::Array2;


// ---------- internal helpers, NOT exposed to Python ----------

/// Daubechies 8 wavelet filters, as separable pairs (vertical, horizontal).
///
/// The 2D filters are outer products: LH = l h^T, HL = h l^T, HH = h h^T.
fn daubechies8() -> [(Vec<f64>, Vec<f64>); 3] {
    let hpdf: [f64; 16] = [
        -0.0544158422,  0.3128715909, -0.6756307363,  0.5853546837,
         0.0158291053, -0.2840155430, -0.0004724846,  0.1287474266,
         0.0173693010, -0.0440882539, -0.0139810279,  0.0087460940,
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
    ];

    // build lpdf
    let mut lpdf = [0f64; 16];
    for i in 0..16 {
        lpdf[i] = ((-1f64).powi(i as i32)) * hpdf[15 - i];
    }

    [
        (lpdf.to_vec(), hpdf.to_vec()),
        (hpdf.to_vec(), lpdf.to_vec()),
        (hpdf.to_vec(), hpdf.to_vec()),
    ]
}

/// Mirror index into [0, n), edge included (SciPy boundary='symm').
fn reflect_index(i: isize, n: isize) -> usize {
    (if i < 0 { -i - 1 } else if i < n { i } else { 2 * n - i - 1 }) as usize
}

/// Symmetric padding of a row-major image of shape [h, w].
fn pad_symmetric(input: &[f64], h: usize, w: usize, pad: usize) -> (Vec<f64>, usize, usize) {
    let (hp, wp) = (h + 2 * pad, w + 2 * pad);
    let cols: Vec<usize> = (0..wp)
        .map(|j| reflect_index(j as isize - pad as isize, w as isize))
        .collect();
    let mut output = vec![0f64; hp * wp];
    for i in 0..hp {
        let ii = reflect_index(i as isize - pad as isize, h as isize);
        let src = &input[ii * w..(ii + 1) * w];
        for (dst, &jj) in output[i * wp..(i + 1) * wp].iter_mut().zip(&cols) {
            *dst = src[jj];
        }
    }
    (output, hp, wp)
}

/// 2D convolution with the separable kernel a b^T, mode='valid'.
///
/// out[i, j] = sum_{u, v} input[i + u, j + v] * a[ka-1-u] * b[kb-1-v]
fn convolve_separable(input: &[f64], h: usize, w: usize, a: &[f64], b: &[f64]) -> (Vec<f64>, usize, usize) {
    let (ka, kb) = (a.len(), b.len());
    let (ho, wo) = (h - ka + 1, w - kb + 1);

    // vertical pass, accumulating whole rows
    let mut tmp = vec![0f64; ho * w];
    for i in 0..ho {
        let dst = &mut tmp[i * w..(i + 1) * w];
        for u in 0..ka {
            let c = a[ka - 1 - u];
            let src = &input[(i + u) * w..(i + u + 1) * w];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += c * s;
            }
        }
    }

    // horizontal pass, flipped kernel
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();
    let mut output = vec![0f64; ho * wo];
    for i in 0..ho {
        let src = &tmp[i * w..(i + 1) * w];
        for (j, d) in output[i * wo..(i + 1) * wo].iter_mut().enumerate() {
            *d = src[j..j + kb].iter().zip(&b_rev).map(|(s, c)| s * c).sum();
        }
    }
    (output, ho, wo)
}

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
