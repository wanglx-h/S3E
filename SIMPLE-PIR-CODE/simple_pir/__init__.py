"""
SimplePIR: Simple and Fast Private Information Retrieval Library

A Python implementation of SimplePIR based on the paper:
"One Server for the Price of Two: Simple and Fast Single-Server Private Information Retrieval"

This library provides efficient private information retrieval capabilities using
Learning with Errors (LWE) based cryptographic techniques.
"""

from .core.pir_server import SimplePIRServer
from .core.pir_client import SimplePIRClient
from .core.pir_protocol import SimplePIRProtocol
from .config.pir_config import SimplePIRConfig, SecurityLevel
from .utils.crypto_utils import LWEParameters, generate_lwe_keys
from .utils.matrix_utils import optimal_matrix_dimensions

__version__ = "1.0.0"
__author__ = "CASE-SSE Research Team"

__all__ = [
    # Core PIR components
    "SimplePIRServer",
    "SimplePIRClient", 
    "SimplePIRProtocol",
    
    # Configuration
    "SimplePIRConfig",
    "SecurityLevel",
    
    # Utilities
    "LWEParameters",
    "generate_lwe_keys",
    "optimal_matrix_dimensions",
    
    # Version info
    "__version__",
    "__author__",
]

# Library metadata
LIBRARY_INFO = {
    "name": "SimplePIR",
    "version": __version__,
    "description": "Simple and Fast Private Information Retrieval Library",
    "paper": "One Server for the Price of Two: Simple and Fast Single-Server Private Information Retrieval",
    "security_model": "Learning with Errors (LWE)",
    "complexity": "O(√n) communication, O(n) computation",
}