#!/usr/bin/env python3
"""
SimplePIR Library Setup Script
Installation script for the SimplePIR (Simple Private Information Retrieval) library
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simple-pir",
    version="1.0.0",
    author="CASE-SSE Research Team",
    author_email="research@example.com",
    description="A fast and simple Private Information Retrieval (PIR) library based on Learning with Errors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/simple-pir",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
        ],
        "benchmark": [
            "matplotlib>=3.3.0",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "simple-pir=simple_pir.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "simple_pir": ["config/*.json", "examples/*.py"],
    },
)