// src/comparisons.rs
use pyo3::prelude::*;

/// 简单的浮点比较：a < b
#[pyfunction]
fn lt_const(a: f64, b: f64) -> PyResult<bool> {
    Ok(a < b)
}

/// 一个非常简化且非安全的 bitwise 比较占位（这里只做 a<b）
#[pyfunction]
fn lt_bits(a_bits: Vec<i64>, b_bits: Vec<i64>) -> PyResult<bool> {
    // 这里 a_bits/b_bits 假设为整数形式（占位），直接比较
    // 真实实现应使用逐位比较协议
    let a_val: i64 = a_bits.iter().fold(0, |acc, &x| acc*2 + x);
    let b_val: i64 = b_bits.iter().fold(0, |acc, &x| acc*2 + x);
    Ok(a_val < b_val)
}

/// 将函数注册到模块中
pub fn init_comparisons(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lt_const, m)?)?;
    m.add_function(wrap_pyfunction!(lt_bits, m)?)?;
    Ok(())
}
