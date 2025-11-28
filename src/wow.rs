
use pyo3::prelude::*;
// use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use numpy::{ndarray::Array, ndarray::Array1, ndarray::Array2, ndarray::Array3, ndarray::stack, ndarray::s, ndarray::Axis};
// use numpy::{PyReadonlyArray2, PyArray2};
// use pyo3::{prelude::*, Python};


// ---------- internal helper, NOT exposed to Python ----------
fn daubechies8() -> (Array3<f64>, Vec<(Array1<f64>, Array1<f64>)>) {
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

    let h = Array::from_shape_vec((16, 1), hpdf.to_vec()).unwrap();
    let l = Array::from_shape_vec((16, 1), lpdf.to_vec()).unwrap();

    // 2D filters (as before)
    let f0 = l.dot(&h.t());
    let f1 = h.dot(&l.t());
    let f2 = h.dot(&h.t());
    let filters = stack(Axis(0), &[f0.view(), f1.view(), f2.view()]).unwrap();

    // 1D separable pairs [(lpdf, hpdf), (hpdf, lpdf), (hpdf, hpdf)]
    let sep = vec![
        (Array1::from_vec(lpdf.to_vec()), Array1::from_vec(hpdf.to_vec())),
        (Array1::from_vec(hpdf.to_vec()), Array1::from_vec(lpdf.to_vec())),
        (Array1::from_vec(hpdf.to_vec()), Array1::from_vec(hpdf.to_vec())),
    ];

    (filters, sep)
}

// Symmetric padding (SciPy boundary='symm')
fn reflect_index(i: isize, n: isize) -> isize {
    if i < 0 { -i-1 }
    else if i < n { i }
    else { 2*n - i - 1 }
}

/// Symmetric pad a 2D array
fn pad_symmetric(input: &Array2<f64>, pad_v: usize, pad_h: usize) -> Array2<f64> {
    let (h, w) = input.dim();
    let mut output = Array2::<f64>::zeros((h + 2*pad_v, w + 2*pad_h));

    for i in 0..output.nrows() {
        for j in 0..output.ncols() {
            let ii = reflect_index(i as isize - pad_v as isize, h as isize);
            let jj = reflect_index(j as isize - pad_h as isize, w as isize);
            output[[i, j]] = input[[ii as usize, jj as usize]];
        }
    }
    output
}

/// 2D convolution with symmetric padding and mode='same'
fn convolve2d(input: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let (h, w) = input.dim();
    let (kh, kw) = kernel.dim();
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let pad = pad_h.max(pad_w);
    let input_pad = pad_symmetric(input, pad_h, pad_w);

    let mut output = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let mut sum = 0.0f64;
            for u in 0..kh {
                for v in 0..kw {
                    let x = i + u;
                    let y = j + v;
                    sum += input_pad[[x, y]] * kernel[[kh - 1 - u, kw - 1 - v]];  // flip kernel
                }
            }
            output[[i, j]] = sum;
        }
    }

    output
}



fn convolve1d_horizontal(input: &Array2<f64>, kernel: &[f64]) -> Array2<f64> {
    let (h, w) = input.dim();
    let k = kernel.len();
    let pad = k / 2;
    let input_pad = pad_symmetric(input, 0, pad);

    let mut out = Array2::<f64>::zeros((h, w));

    for i in 0..h {
        for j in 0..w {
            let mut sum = 0.0;
            for u in 0..k {
                sum += input_pad[[i, j + u]] * kernel[k - 1 - u];
            }
            out[[i, j]] = sum;
        }
    }

    out
}

fn convolve1d_vertical(input: &Array2<f64>, kernel: &[f64]) -> Array2<f64> {
    // transpose the input
    let input_t = input.t();
    let mut tmp = convolve1d_horizontal(&input_t.to_owned(), kernel);
    tmp.t().to_owned() // transpose back
}

// Computes WOW cost.
//
// Parameters
// ----------
// x0 : np.ndarray
//     uncompressed (pixel) cover image of shape [height, width]
//
// Returns
// -------
// np.ndarray
//     cost for +-1 change of shape [height, width]
// #[pyfunction]
// #[pyo3(signature = (x0))]
#[pyfunction]
#[pyo3(signature = (x0, p = -1.0))]
fn compute_cost<'py>(py: Python<'py>, x0: PyReadonlyArray2<'py, u8>, p: f64)
    -> PyResult<Py<PyArray2<f64>>> {

    let input = x0.as_array().mapv(|v| v as f64);
    let (h, w) = input.dim();
    let mut x0_pad = pad_symmetric(&input, 16 as usize, 16 as usize);

    //
    // let filters = daubechies8();
    let (filters, sep_filters) = daubechies8();
    let mut filters_rot = filters.clone();
    filters_rot.invert_axis(Axis(2));
    filters_rot.invert_axis(Axis(1));

    let mut xi = Vec::with_capacity(3);

    for f in 0..3 {

        // // --- 1D separated ---
        // // separable filters: (a, b)
        // let (a, b) = (&sep_filters[f].0, &sep_filters[f].1);
        // // residual
        // let tmp = convolve1d_vertical(&x0_pad, a.as_slice().unwrap());
        // let r  = convolve1d_horizontal(&tmp, b.as_slice().unwrap());
        // // rotate 180 + absolute kernel
        // let mut a_rev = a.clone();
        // a_rev.invert_axis(Axis(0));
        // a_rev = a_rev.to_owned().mapv(|v| v.abs());
        // let mut b_rev = b.clone();
        // b_rev.invert_axis(Axis(0));
        // b_rev = b_rev.to_owned().mapv(|v| v.abs());
        // // suitability
        // let tmp2 = convolve1d_vertical(&r.mapv(|v| v.abs()), a_rev.as_slice().unwrap());
        // let x = convolve1d_horizontal(&tmp2, b_rev.as_slice().unwrap());


        // --- 2D ---
        // filter
        let kernel = filters.index_axis(Axis(0), f).to_owned();
        // residual
        let r = convolve2d(&x0_pad, &kernel);
        // rotate 180 + absolute kernel
        let kernel_rot_abs = filters_rot.index_axis(Axis(0), f).to_owned().mapv(|v| v.abs());
        // suitability
        let x = convolve2d(&r.mapv(|v| v.abs()), &kernel_rot_abs);


        // ----------
        // remove symmetric padding (center crop)
        let crop_h = (x.shape()[0] - h) / 2 + 1;
        let crop_w = (x.shape()[1] - w) / 2 + 1;
        let x_crop = x.slice(s![crop_h..crop_h+h, crop_w..crop_w+w]).to_owned();

        xi.push(x_crop);
    }

    // convert xi Vec<Array2<f64>> into a single Array3<f64> of shape (3, h, w)
    let xi_3d = Array3::from_shape_vec(
        (3, h, w),
        xi.into_iter().flat_map(|arr| arr.into_raw_vec()).collect()
    ).unwrap();

    // compute sum over channels of xi_i^p
    let rho = xi_3d.mapv(|v| v.max(f64::EPSILON)).mapv(|v| v.powf(p)).sum_axis(Axis(0)).mapv(|v| v.powf(-1.0f64 / p));
    Ok(PyArray2::from_owned_array(py, rho).into())
}

#[pymodule]
pub fn init_wow_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cost, m)?)?;
    Ok(())
}