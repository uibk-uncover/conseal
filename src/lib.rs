
use pyo3::prelude::*;
// use pyo3::types::PyDict;
// use numpy::{PyArray1, PyReadonlyArray2};

mod hill;
mod juniward;
mod wpc;
mod wpc_lt;
mod suniward;
mod uerd;
mod wavelet;
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
    //
    let suniward_mod = PyModule::new(py, "suniward")?;
    suniward::init_suniward_module(py, &suniward_mod)?;
    m.add_submodule(&suniward_mod)?;
    //
    let juniward_mod = PyModule::new(py, "juniward")?;
    juniward::init_juniward_module(py, &juniward_mod)?;
    m.add_submodule(&juniward_mod)?;
    //
    let uerd_mod = PyModule::new(py, "uerd")?;
    uerd::init_uerd_module(py, &uerd_mod)?;
    m.add_submodule(&uerd_mod)?;
    //
    let lt_mod = PyModule::new(py, "lt")?;
    wpc_lt::init_lt_module(py, &lt_mod)?;
    m.add_submodule(&lt_mod)?;
    //
    let wpc_mod = PyModule::new(py, "wpc")?;
    wpc::init_wpc_module(py, &wpc_mod)?;
    m.add_submodule(&wpc_mod)?;

    // m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
