use pyo3::prelude::*;
use numpy::{PyArray4, PyReadonlyArray2, PyReadonlyArray4};
use numpy::ndarray::Array4;

use crate::wavelet::reflect_index;


// Computes UERD cost.
//
// Parameters
// ----------
// y0 : np.ndarray
//     quantized cover DCT coefficients (int16, C-contiguous)
//     of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
// qt : np.ndarray
//     quantization table (uint16) of shape [8, 8]
//
// Returns
// -------
// np.ndarray
//     cost for +-1 change of shape [num_vertical_blocks, num_horizontal_blocks, 8, 8]
#[pyfunction]
fn compute_cost<'py>(
    py: Python<'py>,
    y0: PyReadonlyArray4<'py, i16>,
    qt: PyReadonlyArray2<'py, u16>,
) -> PyResult<Py<PyArray4<f64>>> {

    let (nby, nbx, _, _) = y0.as_array().dim();
    let y0 = y0.as_slice()?;
    let qt: Vec<f64> = qt.as_array().iter().map(|&q| q as f64).collect();

    // block energies, Eq. 3: sum of dequantized AC magnitudes
    let energy: Vec<f64> = y0
        .chunks_exact(64)
        .map(|block| {
            block[1..].iter().zip(&qt[1..]).map(|(&y, &q)| (y as f64 * q).abs()).sum()
        })
        .collect();

    // block cost: energy + 1/4 of the energy in the 8-neighborhood (symmetric boundary)
    let mut block_rho = vec![0f64; nby * nbx];
    for m in 0..nby {
        for n in 0..nbx {
            let mut neighborhood = 0f64;
            for dm in -1..=1isize {
                for dn in -1..=1isize {
                    if dm == 0 && dn == 0 {
                        continue;
                    }
                    let mm = reflect_index(m as isize + dm, nby as isize);
                    let nn = reflect_index(n as isize + dn, nbx as isize);
                    neighborhood += energy[mm * nbx + nn];
                }
            }
            block_rho[m * nbx + n] = energy[m * nbx + n] + 0.25 * neighborhood;
        }
    }

    // mode cost: quantization step, DC as the average of its AC neighbors
    let mut mode_rho = [0f64; 64];
    mode_rho.copy_from_slice(&qt);
    mode_rho[0] = 0.5 * (qt[1] + qt[8]);

    // cost, Eq. 4, infinite for zero block cost
    let mut rho = vec![0f64; nby * nbx * 64];
    for (b, block) in rho.chunks_mut(64).enumerate() {
        let br = block_rho[b];
        for (r, &mr) in block.iter_mut().zip(&mode_rho) {
            *r = if br > 0.0 { mr / br } else { f64::INFINITY };
        }
    }
    let rho = Array4::from_shape_vec((nby, nbx, 8, 8), rho).unwrap();
    Ok(PyArray4::from_owned_array(py, rho).into())
}

#[pymodule]
pub fn init_uerd_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cost, m)?)?;
    Ok(())
}
