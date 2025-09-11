
use pyo3::prelude::*;
// use pyo3::types::PyDict;
use numpy::{PyArray2, PyReadonlyArray2, ndarray::Array};


/// Computes HILL cost.
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
fn compute_cost<'py>(py: Python<'py>, x0: PyReadonlyArray2<'py, u8>) -> PyResult<Py<PyArray2<f32>>> {
    let input = x0.as_array();
    let (h, w) = input.dim();
    let mut x0_pad = Array::<f32, _>::ones((h + 2*9, w + 2*9));

    // pad array
    for row in 0..x0_pad.nrows() {
        for col in 0..x0_pad.ncols() {
            let mut rr = row as isize - 9 as isize;
            let mut cc = col as isize - 9 as isize;
            // reflect row index
            if rr < 0 {
                rr = -rr - 1;
            }
            if rr >= h as isize {
                rr = 2 * h as isize - rr - 1;
            }
            // reflect col index
            if cc < 0 {
                cc = -cc - 1;
            }
            if cc >= w as isize {
                cc = 2 * w as isize - cc - 1;
            }
            //
            x0_pad[[row, col]] = input[[rr as usize, cc as usize]] as f32;
        }
    }

    // convolve with KB
    let mut I1 = Array::<f32, _>::zeros((h + 2*8, w + 2*8));
    for i in 1..x0_pad.nrows()-1 {
        for j in 1..x0_pad.ncols()-1 {
            let val =
                -1.0*x0_pad[[i-1, j-1]]+2.0*x0_pad[[i-1, j+0]]-1.0*x0_pad[[i-1, j+1]]
                +2.0*x0_pad[[i+0, j-1]]-4.0*x0_pad[[i+0, j+0]]+2.0*x0_pad[[i+0, j+1]]
                -1.0*x0_pad[[i+1, j-1]]+2.0*x0_pad[[i+1, j+0]]-1.0*x0_pad[[i+1, j+1]];

            I1[[i-1, j-1]] = (val / 4.0f32).abs();
        }
    }

    // convolve with AVG 3x3
    let mut I2 = Array::<f32, _>::zeros((h + 2*7, w + 2*7));
    for i in 1..I1.nrows()-1 {
        for j in 1..I1.ncols()-1 {
            let val =
                I1[[i-1, j-1]]+I1[[i-1, j+0]]+I1[[i-1, j+1]]+
                I1[[i+0, j-1]]+I1[[i+0, j+0]]+I1[[i+0, j+1]]+
                I1[[i+1, j-1]]+I1[[i+1, j+0]]+I1[[i+1, j+1]];

            I2[[i-1, j-1]] = 1.0f32 / (val / 9.0f32).max(f32::EPSILON);
        }
    }

    // convolve with AVG 15x15
    let mut cost = Array::<f32, _>::zeros((h, w));
    for i in 7..I2.nrows()-7 {
        for j in 7..I2.ncols()-7 {
            let val =
                I2[[i-7, j-7]]+I2[[i-7, j-6]]+I2[[i-7, j-5]]+I2[[i-7, j-4]]+I2[[i-7, j-3]]+I2[[i-7, j-2]]+I2[[i-7, j-1]]+I2[[i-7, j+0]]+I2[[i-7, j+1]]+I2[[i-7, j+2]]+I2[[i-7, j+3]]+I2[[i-7, j+4]]+I2[[i-7, j+5]]+I2[[i-7, j+6]]+I2[[i-7, j+7]]+
                I2[[i-6, j-7]]+I2[[i-6, j-6]]+I2[[i-6, j-5]]+I2[[i-6, j-4]]+I2[[i-6, j-3]]+I2[[i-6, j-2]]+I2[[i-6, j-1]]+I2[[i-6, j+0]]+I2[[i-6, j+1]]+I2[[i-6, j+2]]+I2[[i-6, j+3]]+I2[[i-6, j+4]]+I2[[i-6, j+5]]+I2[[i-6, j+6]]+I2[[i-6, j+7]]+
                I2[[i-5, j-7]]+I2[[i-5, j-6]]+I2[[i-5, j-5]]+I2[[i-5, j-4]]+I2[[i-5, j-3]]+I2[[i-5, j-2]]+I2[[i-5, j-1]]+I2[[i-5, j+0]]+I2[[i-5, j+1]]+I2[[i-5, j+2]]+I2[[i-5, j+3]]+I2[[i-5, j+4]]+I2[[i-5, j+5]]+I2[[i-5, j+6]]+I2[[i-5, j+7]]+
                I2[[i-4, j-7]]+I2[[i-4, j-6]]+I2[[i-4, j-5]]+I2[[i-4, j-4]]+I2[[i-4, j-3]]+I2[[i-4, j-2]]+I2[[i-4, j-1]]+I2[[i-4, j+0]]+I2[[i-4, j+1]]+I2[[i-4, j+2]]+I2[[i-4, j+3]]+I2[[i-4, j+4]]+I2[[i-4, j+5]]+I2[[i-4, j+6]]+I2[[i-4, j+7]]+
                I2[[i-3, j-7]]+I2[[i-3, j-6]]+I2[[i-3, j-5]]+I2[[i-3, j-4]]+I2[[i-3, j-3]]+I2[[i-3, j-2]]+I2[[i-3, j-1]]+I2[[i-3, j+0]]+I2[[i-3, j+1]]+I2[[i-3, j+2]]+I2[[i-3, j+3]]+I2[[i-3, j+4]]+I2[[i-3, j+5]]+I2[[i-3, j+6]]+I2[[i-3, j+7]]+
                I2[[i-2, j-7]]+I2[[i-2, j-6]]+I2[[i-2, j-5]]+I2[[i-2, j-4]]+I2[[i-2, j-3]]+I2[[i-2, j-2]]+I2[[i-2, j-1]]+I2[[i-2, j+0]]+I2[[i-2, j+1]]+I2[[i-2, j+2]]+I2[[i-2, j+3]]+I2[[i-2, j+4]]+I2[[i-2, j+5]]+I2[[i-2, j+6]]+I2[[i-2, j+7]]+
                I2[[i-1, j-7]]+I2[[i-1, j-6]]+I2[[i-1, j-5]]+I2[[i-1, j-4]]+I2[[i-1, j-3]]+I2[[i-1, j-2]]+I2[[i-1, j-1]]+I2[[i-1, j+0]]+I2[[i-1, j+1]]+I2[[i-1, j+2]]+I2[[i-1, j+3]]+I2[[i-1, j+4]]+I2[[i-1, j+5]]+I2[[i-1, j+6]]+I2[[i-1, j+7]]+
                I2[[i+0, j-7]]+I2[[i+0, j-6]]+I2[[i+0, j-5]]+I2[[i+0, j-4]]+I2[[i+0, j-3]]+I2[[i+0, j-2]]+I2[[i+0, j-1]]+I2[[i+0, j+0]]+I2[[i+0, j+1]]+I2[[i+0, j+2]]+I2[[i+0, j+3]]+I2[[i+0, j+4]]+I2[[i+0, j+5]]+I2[[i+0, j+6]]+I2[[i+0, j+7]]+
                I2[[i+1, j-7]]+I2[[i+1, j-6]]+I2[[i+1, j-5]]+I2[[i+1, j-4]]+I2[[i+1, j-3]]+I2[[i+1, j-2]]+I2[[i+1, j-1]]+I2[[i+1, j+0]]+I2[[i+1, j+1]]+I2[[i+1, j+2]]+I2[[i+1, j+3]]+I2[[i+1, j+4]]+I2[[i+1, j+5]]+I2[[i+1, j+6]]+I2[[i+1, j+7]]+
                I2[[i+2, j-7]]+I2[[i+2, j-6]]+I2[[i+2, j-5]]+I2[[i+2, j-4]]+I2[[i+2, j-3]]+I2[[i+2, j-2]]+I2[[i+2, j-1]]+I2[[i+2, j+0]]+I2[[i+2, j+1]]+I2[[i+2, j+2]]+I2[[i+2, j+3]]+I2[[i+2, j+4]]+I2[[i+2, j+5]]+I2[[i+2, j+6]]+I2[[i+2, j+7]]+
                I2[[i+3, j-7]]+I2[[i+3, j-6]]+I2[[i+3, j-5]]+I2[[i+3, j-4]]+I2[[i+3, j-3]]+I2[[i+3, j-2]]+I2[[i+3, j-1]]+I2[[i+3, j+0]]+I2[[i+3, j+1]]+I2[[i+3, j+2]]+I2[[i+3, j+3]]+I2[[i+3, j+4]]+I2[[i+3, j+5]]+I2[[i+3, j+6]]+I2[[i+3, j+7]]+
                I2[[i+4, j-7]]+I2[[i+4, j-6]]+I2[[i+4, j-5]]+I2[[i+4, j-4]]+I2[[i+4, j-3]]+I2[[i+4, j-2]]+I2[[i+4, j-1]]+I2[[i+4, j+0]]+I2[[i+4, j+1]]+I2[[i+4, j+2]]+I2[[i+4, j+3]]+I2[[i+4, j+4]]+I2[[i+4, j+5]]+I2[[i+4, j+6]]+I2[[i+4, j+7]]+
                I2[[i+5, j-7]]+I2[[i+5, j-6]]+I2[[i+5, j-5]]+I2[[i+5, j-4]]+I2[[i+5, j-3]]+I2[[i+5, j-2]]+I2[[i+5, j-1]]+I2[[i+5, j+0]]+I2[[i+5, j+1]]+I2[[i+5, j+2]]+I2[[i+5, j+3]]+I2[[i+5, j+4]]+I2[[i+5, j+5]]+I2[[i+5, j+6]]+I2[[i+5, j+7]]+
                I2[[i+6, j-7]]+I2[[i+6, j-6]]+I2[[i+6, j-5]]+I2[[i+6, j-4]]+I2[[i+6, j-3]]+I2[[i+6, j-2]]+I2[[i+6, j-1]]+I2[[i+6, j+0]]+I2[[i+6, j+1]]+I2[[i+6, j+2]]+I2[[i+6, j+3]]+I2[[i+6, j+4]]+I2[[i+6, j+5]]+I2[[i+6, j+6]]+I2[[i+6, j+7]]+
                I2[[i+7, j-7]]+I2[[i+7, j-6]]+I2[[i+7, j-5]]+I2[[i+7, j-4]]+I2[[i+7, j-3]]+I2[[i+7, j-2]]+I2[[i+7, j-1]]+I2[[i+7, j+0]]+I2[[i+7, j+1]]+I2[[i+7, j+2]]+I2[[i+7, j+3]]+I2[[i+7, j+4]]+I2[[i+7, j+5]]+I2[[i+7, j+6]]+I2[[i+7, j+7]];
            cost[[i-7, j-7]] = val / 225.0f32;
        }
    }

    Ok(PyArray2::from_array(py, &cost).into())
}


#[pymodule]
pub fn init_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cost, m)?)?;
    Ok(())
}