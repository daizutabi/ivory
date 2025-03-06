mod expressions;
use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

#[pymodule]
fn ivory(_py: Python, _m: &Bound<PyModule>) -> PyResult<()> {
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
