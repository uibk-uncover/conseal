
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::wpc_lt;


// ---------- internal helpers, NOT exposed to Python ----------

/// Bounds of num contiguous parts of n items, as numpy.array_split.
///
/// The first n % num parts are one item longer.
fn split_bounds(n: usize, num: usize) -> Vec<usize> {
    let (size, extra) = (n / num, n % num);
    let mut bounds = vec![0usize; num + 1];
    for i in 0..num {
        bounds[i + 1] = bounds[i] + size + (i < extra) as usize;
    }
    bounds
}

/// Bounds of the blocks of elements and of symbols.
fn blocks(n: usize, m: usize, block_size: usize) -> (Vec<usize>, Vec<usize>) {
    let num_blocks = m.div_ceil(block_size);
    (split_bounds(n, num_blocks), split_bounds(m, num_blocks))
}


// ---------- Python interface ----------

/// Embeds message symbols into cover symbols with wet-paper code.
///
/// Block loop of conseal.coding.wpc.embed, see there.
///
/// Parameters
/// ----------
/// x : np.ndarray
///     cover symbols of shape [n], uint8
/// is_dry : np.ndarray
///     whether element is dry, of shape [n], bool
/// rho : np.ndarray
///     cost of a change, of shape [n], float64, cheaper dry elements first
/// symbols : np.ndarray
///     message symbols of shape [m], uint8
/// h : np.ndarray
///     common parity-check matrix of the largest block, uint8
/// block_size : int
///     number of message symbols per block
/// q : int
///     prime field size
///
/// Returns
/// -------
/// np.ndarray
///     changes of shape [n], uint8
#[pyfunction]
#[pyo3(signature = (x, is_dry, rho, symbols, h, block_size, q))]
#[allow(clippy::too_many_arguments)]
fn embed<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, u8>,
    is_dry: PyReadonlyArray1<'py, bool>,
    rho: PyReadonlyArray1<'py, f64>,
    symbols: PyReadonlyArray1<'py, u8>,
    h: PyReadonlyArray2<'py, u8>,
    block_size: usize,
    q: u64,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (x, is_dry, rho, symbols, h) = (
        x.as_array(), is_dry.as_array(), rho.as_array(), symbols.as_array(), h.as_array(),
    );
    let (n, m) = (x.len(), symbols.len());
    let mut changes = vec![0u8; n];
    if m == 0 {
        return Ok(changes.into_pyarray(py));
    }

    // embed block by block
    let (element_bounds, symbol_bounds) = blocks(n, m, block_size);
    for i in 0..element_bounds.len() - 1 {
        let (e0, e1) = (element_bounds[i], element_bounds[i + 1]);
        let (s0, s1) = (symbol_bounds[i], symbol_bounds[i + 1]);
        let p = s1 - s0;
        let weights: Vec<u64> = (0..p as u32).map(|k| q.pow(k)).collect();

        // dry columns of the block, cheaper first, stable
        let mut dry: Vec<usize> = (0..e1 - e0).filter(|&j| is_dry[e0 + j]).collect();
        dry.sort_by(|&a, &b| {
            rho[e0 + a].partial_cmp(&rho[e0 + b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // syndrome difference z = message - H x, as base-q integer
        let s: u64 = (0..p)
            .map(|r| {
                let hx: u64 = (0..e1 - e0).map(|j| h[[r, j]] as u64 * x[e0 + j] as u64).sum();
                (symbols[s0 + r] as u64 + q - hx % q) % q * weights[r]
            })
            .sum();

        // dry columns of H, as base-q integers
        let columns: Vec<u64> = dry
            .iter()
            .map(|&j| (0..p).map(|r| h[[r, j]] as u64 * weights[r]).sum())
            .collect();

        // fewest changes
        let v = wpc_lt::solve(&columns, s, q, p).ok_or_else(|| {
            PyValueError::new_err(format!(
                "block {}: {} symbols do not fit into {} dry elements", i, p, dry.len(),
            ))
        })?;

        // symbol shifts are the changes, back in cover order
        for (k, &j) in dry.iter().enumerate() {
            changes[e0 + j] = v[k];
        }
    }
    Ok(changes.into_pyarray(py))
}

/// Extracts message symbols from stego symbols, embedded with wet-paper code.
///
/// Block loop of conseal.coding.wpc.extract, see there.
///
/// Parameters
/// ----------
/// y : np.ndarray
///     stego symbols of shape [n], uint8
/// m : int
///     number of message symbols
/// h : np.ndarray
///     common parity-check matrix of the largest block, uint8
/// block_size : int
///     number of message symbols per block
/// q : int
///     prime field size
///
/// Returns
/// -------
/// np.ndarray
///     message symbols of shape [m], uint8
#[pyfunction]
#[pyo3(signature = (y, m, h, block_size, q))]
fn extract<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, u8>,
    m: usize,
    h: PyReadonlyArray2<'py, u8>,
    block_size: usize,
    q: u64,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (y, h) = (y.as_array(), h.as_array());
    let mut symbols = vec![0u8; m];
    if m == 0 {
        return Ok(symbols.into_pyarray(py));
    }

    // message is the syndrome H y of each block
    let (element_bounds, symbol_bounds) = blocks(y.len(), m, block_size);
    for i in 0..element_bounds.len() - 1 {
        let (e0, e1) = (element_bounds[i], element_bounds[i + 1]);
        let (s0, s1) = (symbol_bounds[i], symbol_bounds[i + 1]);
        for r in 0..s1 - s0 {
            let hy: u64 = (0..e1 - e0).map(|j| h[[r, j]] as u64 * y[e0 + j] as u64).sum();
            symbols[s0 + r] = (hy % q) as u8;
        }
    }
    Ok(symbols.into_pyarray(py))
}

pub fn init_wpc_module(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed, m)?)?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
