
use pyo3::prelude::*;
// use pyo3::types::PyDict;
// use numpy::{PyArray1, PyReadonlyArray2};

mod hill;
mod wow;

#[pymodule]
fn _conseal(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
	//
    let hill_mod = PyModule::new(py, "hill")?;
    hill::init_hill_module(py, &hill_mod)?;
    m.add_submodule(&hill_mod)?;
    //
	let wow_mod = PyModule::new(py, "wow")?;
    wow::init_wow_module(py, &wow_mod)?;
    m.add_submodule(&wow_mod)?;

    // m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
