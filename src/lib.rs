
use pyo3::prelude::*;
// use pyo3::types::PyDict;
// use numpy::{PyArray1, PyReadonlyArray2};

mod hill;

#[pymodule]
fn _conseal(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
	let hill_mod = PyModule::new(py, "hill")?;
    hill::init_module(py, &hill_mod)?;
    m.add_submodule(&hill_mod)?;

    // m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
