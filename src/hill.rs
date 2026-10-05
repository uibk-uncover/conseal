
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use numpy::ndarray::Array2;


// ---------- internal helpers, NOT exposed to Python ----------

/// Mirror index into [0, n), edge included (SciPy boundary='symm').
fn reflect_index(i: isize, n: isize) -> usize {
    (if i < 0 { -i - 1 } else if i < n { i } else { 2 * n - i - 1 }) as usize
}

/// Symmetric padding of a row-major image of shape [h, w].
///
/// All HILL filters are symmetric, so padding the image once
/// equals SciPy's symmetric padding at every filtering stage.
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

/// Computes HILL cost.
///
/// Computed in float64, 15x15 low-pass separable.
///
/// Parameters
/// ----------
/// x0 : np.ndarray
///     uncompressed (pixel) cover image of shape [height, width]
///
/// Returns
/// -------
/// np.ndarray
///     cost for +-1 change of shape [height, width]
#[pyfunction]
#[pyo3(signature = (x0))]
fn compute_cost<'py>(py: Python<'py>, x0: PyReadonlyArray2<'py, u8>) -> PyResult<Py<PyArray2<f64>>> {
    let x0 = x0.as_array();
    let (h, w) = x0.dim();
    let input: Vec<f64> = x0.iter().map(|&v| v as f64).collect();

    // pad once, all the convolutions are 'valid'
    let (x_pad, hp, wp) = pad_symmetric(&input, h, w, 9);

    // high-pass filter KB, |I1 / 4|
    let (h1, w1) = (hp - 2, wp - 2);
    let mut i1 = vec![0f64; h1 * w1];
    for i in 0..h1 {
        let r0 = &x_pad[i * wp..(i + 1) * wp];
        let r1 = &x_pad[(i + 1) * wp..(i + 2) * wp];
        let r2 = &x_pad[(i + 2) * wp..(i + 3) * wp];
        for (j, d) in i1[i * w1..(i + 1) * w1].iter_mut().enumerate() {
            let val =
                -1.0 * r0[j] + 2.0 * r0[j + 1] - 1.0 * r0[j + 2]
                + 2.0 * r1[j] - 4.0 * r1[j + 1] + 2.0 * r1[j + 2]
                - 1.0 * r2[j] + 2.0 * r2[j + 1] - 1.0 * r2[j + 2];
            *d = (val / 4.0f64).abs();
        }
    }

    // low-pass filter 3x3, then reciprocal of the clipped value
    let l1 = 1.0f64 / 9.0f64;
    let (h2, w2) = (h1 - 2, w1 - 2);
    let mut i2 = vec![0f64; h2 * w2];
    for i in 0..h2 {
        let r0 = &i1[i * w1..(i + 1) * w1];
        let r1 = &i1[(i + 1) * w1..(i + 2) * w1];
        let r2 = &i1[(i + 2) * w1..(i + 3) * w1];
        for (j, d) in i2[i * w2..(i + 1) * w2].iter_mut().enumerate() {
            let val =
                r0[j] * l1 + r0[j + 1] * l1 + r0[j + 2] * l1
                + r1[j] * l1 + r1[j + 1] * l1 + r1[j + 2] * l1
                + r2[j] * l1 + r2[j + 1] * l1 + r2[j + 2] * l1;
            *d = 1.0f64 / val.max(f32::EPSILON as f64);
        }
    }

    // low-pass filter 15x15, separable
    let l2 = 1.0f64 / 15.0f64;
    // vertical pass, accumulating whole rows
    let mut tmp = vec![0f64; h * w2];
    for i in 0..h {
        let dst = &mut tmp[i * w2..(i + 1) * w2];
        for u in 0..15 {
            let src = &i2[(i + u) * w2..(i + u + 1) * w2];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += s * l2;
            }
        }
    }
    // horizontal pass
    let mut cost = vec![0f64; h * w];
    for i in 0..h {
        let src = &tmp[i * w2..(i + 1) * w2];
        for (j, d) in cost[i * w..(i + 1) * w].iter_mut().enumerate() {
            *d = src[j..j + 15].iter().map(|&s| s * l2).sum();
        }
    }

    let cost = Array2::from_shape_vec((h, w), cost).unwrap();
    Ok(PyArray2::from_owned_array(py, cost).into())
}

#[pymodule]
pub fn init_hill_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cost, m)?)?;
    Ok(())
}
