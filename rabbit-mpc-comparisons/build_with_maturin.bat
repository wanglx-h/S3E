@echo off
REM build_with_maturin.bat
REM 在 Windows 上使用 maturin 构建并在当前 Python 环境中安装 rabbit 模块
REM 使用前请确保已安装：Rust toolchain（rustup + cargo）、Microsoft Build Tools (C++), Python dev headers

REM 1) 激活你的 Python 虚拟环境（如果有的话）
REM call C:\path\to\venv\Scripts\activate.bat

REM 2) 安装 maturin（如果尚未安装）
python -m pip install --upgrade pip
python -m pip install maturin

REM 3) 可选：确保 Rust 编译工具链存在
rustup --version >nul 2>&1
if errorlevel 1 (
    echo Rust toolchain not found. Please install rustup from https://rustup.rs/
    pause
    exit /b 1
)

REM 4) 运行 maturin 开发安装（等价于 pip install -e .）
maturin develop --release --strip

if errorlevel 1 (
    echo maturin build failed. Check the error messages above.
    pause
    exit /b 1
)

echo.
echo Build succeeded. You should be able to `python -c "from rabbit import comparisons; print(comparisons.lt_const(1,2))"`
pause
