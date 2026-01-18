// src/lib.rs
use pyo3::prelude::*;

mod comparisons; // 我们将在 comparisons.rs 中导出函数

#[pymodule]
fn rabbit(_py: Python, m: &PyModule) -> PyResult<()> {
    // 将 Rust 的 comparisons 模块注册为 rabbit.comparisons 子模块
    let submodule = PyModule::new(_py, "comparisons")?;
    comparisons::init_comparisons(_py, submodule)?;
    m.add_submodule(submodule)?;
    Ok(())
}
