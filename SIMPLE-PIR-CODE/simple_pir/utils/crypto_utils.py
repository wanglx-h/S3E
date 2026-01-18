"""
Cryptographic utilities for SimplePIR
LWE-based cryptographic operations and key generation
"""

import numpy as np
import hashlib
import time
from typing import Dict, Any, Tuple, List
import secrets


class LWEParameters:
    """Learning with Errors (LWE) parameters for SimplePIR"""
    
    def __init__(self, params_dict: Dict[str, Any]):
        """
        Initialize LWE parameters
        
        Args:
            params_dict: Dictionary containing LWE parameters
        """
        self.security_parameter = params_dict["security_parameter"]
        self.modulus = params_dict["modulus"]
        self.noise_bound = params_dict["noise_bound"]
        self.dimension = params_dict.get("dimension", self.security_parameter)
        self.error_distribution = params_dict.get("error_distribution", "gaussian")
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate LWE parameters for security and correctness"""
        if self.security_parameter < 80:
            raise ValueError("Security parameter must be at least 80 bits")
        
        if not self._is_power_of_two_or_prime(self.modulus):
            print(f"Warning: Modulus {self.modulus} should be prime or power of 2 for optimal security")
        
        if self.noise_bound >= self.modulus // 4:
            print(f"Warning: Noise bound {self.noise_bound} may be too large relative to modulus")
        
        if self.dimension < self.security_parameter // 2:
            print(f"Warning: LWE dimension {self.dimension} may be too small for {self.security_parameter}-bit security")
        elif self.dimension < self.security_parameter:
            # 这是正常情况，论文中n=1024, λ=128，无需警告
            pass
    
    def _is_power_of_two_or_prime(self, n: int) -> bool:
        """Check if number is power of 2 or prime (simplified check)"""
        # Check power of 2
        if n > 0 and (n & (n - 1)) == 0:
            return True
        
        # Basic primality check for small numbers
        if n < 2:
            return False
        for i in range(2, min(int(n**0.5) + 1, 1000)):
            if n % i == 0:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            "security_parameter": self.security_parameter,
            "modulus": self.modulus,
            "noise_bound": self.noise_bound,
            "dimension": self.dimension,
            "error_distribution": self.error_distribution
        }
    
    def __str__(self) -> str:
        return (f"LWE(n={self.dimension}, q={self.modulus}, "
                f"σ={self.noise_bound}, λ={self.security_parameter})")


def generate_lwe_keys(lwe_params: LWEParameters) -> Dict[str, np.ndarray]:
    """
    Generate LWE key pair
    
    Args:
        lwe_params: LWE parameters
        
    Returns:
        Dictionary containing public and private keys
    """
    # Generate secret key
    secret_key = np.random.randint(0, lwe_params.modulus, size=lwe_params.dimension)
    
    # Generate public key matrix A
    public_key_A = np.random.randint(
        0, lwe_params.modulus,
        size=(lwe_params.dimension, lwe_params.dimension)
    )
    
    # Generate error vector
    error_vector = generate_noise(lwe_params.dimension, lwe_params.noise_bound)
    
    # Compute public key vector b = A*s + e
    As = np.dot(public_key_A, secret_key) % lwe_params.modulus
    public_key_b = (As + error_vector) % lwe_params.modulus
    
    return {
        "secret_key": secret_key,
        "public_key_A": public_key_A,
        "public_key_b": public_key_b,
        "generation_time": time.time()
    }


def generate_noise(size: int, noise_bound: int, 
                  distribution: str = "gaussian") -> np.ndarray:
    """
    Generate noise vector for LWE
    
    Args:
        size: Size of noise vector
        noise_bound: Maximum absolute value of noise
        distribution: Type of noise distribution
        
    Returns:
        Noise vector
    """
    if distribution == "gaussian":
        # Discrete Gaussian noise (approximated)
        sigma = noise_bound / 3.0  # 3-sigma rule
        noise = np.random.normal(0, sigma, size)
        noise = np.round(noise).astype(int)
        noise = np.clip(noise, -noise_bound, noise_bound)
        
    elif distribution == "uniform":
        # Uniform noise in [-noise_bound, noise_bound]
        noise = np.random.randint(-noise_bound, noise_bound + 1, size=size)
        
    elif distribution == "binary":
        # Binary noise {-1, 0, 1}
        noise = np.random.choice([-1, 0, 1], size=size)
        
    else:
        raise ValueError(f"Unsupported noise distribution: {distribution}")
    
    return noise


def lwe_encrypt(message: int, public_key_A: np.ndarray, public_key_b: np.ndarray,
               lwe_params: LWEParameters) -> Tuple[np.ndarray, int]:
    """
    Encrypt message using LWE encryption
    
    Args:
        message: Message to encrypt (0 or 1)
        public_key_A: Public key matrix A
        public_key_b: Public key vector b
        lwe_params: LWE parameters
        
    Returns:
        LWE ciphertext (a, b)
    """
    # Generate random vector r
    r = np.random.randint(0, 2, size=len(public_key_b))
    
    # Compute ciphertext
    # a = A^T * r
    a = np.dot(public_key_A.T, r) % lwe_params.modulus
    
    # b = b^T * r + message * floor(q/2) + error
    b_inner = np.dot(public_key_b, r) % lwe_params.modulus
    message_term = message * (lwe_params.modulus // 2)
    error = generate_noise(1, lwe_params.noise_bound)[0]
    b = (b_inner + message_term + error) % lwe_params.modulus
    
    return a, b


def lwe_decrypt(ciphertext: Tuple[np.ndarray, int], secret_key: np.ndarray,
               lwe_params: LWEParameters) -> int:
    """
    Decrypt LWE ciphertext
    
    Args:
        ciphertext: LWE ciphertext (a, b)
        secret_key: Secret key
        lwe_params: LWE parameters
        
    Returns:
        Decrypted message (0 or 1)
    """
    a, b = ciphertext
    
    # Compute b - <a, s>
    inner_product = np.dot(a, secret_key) % lwe_params.modulus
    decryption_value = (b - inner_product) % lwe_params.modulus
    
    # Determine message by rounding
    threshold = lwe_params.modulus // 4
    if decryption_value < threshold or decryption_value > lwe_params.modulus - threshold:
        return 0
    else:
        return 1


def compute_inner_product(vec1: np.ndarray, vec2: np.ndarray, 
                         modulus: int) -> int:
    """
    Compute inner product modulo q
    
    Args:
        vec1: First vector
        vec2: Second vector
        modulus: Modulus for computation
        
    Returns:
        Inner product mod q
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    
    result = 0
    for a, b in zip(vec1, vec2):
        result += (int(a) * int(b)) % modulus
        result %= modulus
    
    return result


def hash_to_matrix(data: str, rows: int, cols: int, modulus: int) -> np.ndarray:
    """
    Hash data to generate a matrix of specified dimensions
    
    Args:
        data: Input data to hash
        rows: Number of rows
        cols: Number of columns
        modulus: Modulus for matrix elements
        
    Returns:
        Generated matrix
    """
    # Create hash of data
    hash_bytes = hashlib.sha256(data.encode()).digest()
    
    # Extend hash if needed
    total_elements = rows * cols
    needed_bytes = total_elements * 4  # 4 bytes per int
    
    extended_hash = hash_bytes
    while len(extended_hash) < needed_bytes:
        extended_hash += hashlib.sha256(extended_hash).digest()
    
    # Convert to matrix
    matrix = np.zeros((rows, cols), dtype=int)
    byte_idx = 0
    
    for i in range(rows):
        for j in range(cols):
            # Convert 4 bytes to int
            int_bytes = extended_hash[byte_idx:byte_idx+4]
            matrix[i, j] = int.from_bytes(int_bytes, 'big') % modulus
            byte_idx += 4
    
    return matrix


def secure_random_matrix(rows: int, cols: int, modulus: int) -> np.ndarray:
    """
    Generate cryptographically secure random matrix
    
    Args:
        rows: Number of rows
        cols: Number of columns
        modulus: Modulus for matrix elements
        
    Returns:
        Secure random matrix
    """
    matrix = np.zeros((rows, cols), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = secrets.randbelow(modulus)
    
    return matrix


def validate_lwe_security(lwe_params: LWEParameters) -> Dict[str, Any]:
    """
    Validate LWE parameters for security
    
    Args:
        lwe_params: LWE parameters to validate
        
    Returns:
        Security validation report
    """
    security_report = {
        "security_level_bits": lwe_params.security_parameter,
        "parameter_validation": {},
        "recommendations": [],
        "overall_security": "unknown"
    }
    
    # Check dimension vs security parameter ratio
    # SimplePIR论文：n=1024, λ=128, ratio=8.0 为理想比例
    dim_ratio = lwe_params.dimension / lwe_params.security_parameter
    if dim_ratio < 2.0:  # 至少应该是安全参数的2倍
        security_report["recommendations"].append(
            f"Increase LWE dimension (currently {dim_ratio:.2f}x security parameter, recommend ≥4x)"
        )
    
    # Check modulus size
    modulus_bits = lwe_params.modulus.bit_length()
    if modulus_bits < lwe_params.security_parameter:
        security_report["recommendations"].append(
            f"Increase modulus size (currently {modulus_bits} bits)"
        )
    
    # Check noise-to-modulus ratio
    noise_ratio = lwe_params.noise_bound / lwe_params.modulus
    if noise_ratio > 0.25:
        security_report["recommendations"].append(
            f"Reduce noise bound (currently {noise_ratio:.6f} of modulus)"
        )
    elif noise_ratio < 0.00001:  # 提高精度阈值，适配大模数
        security_report["recommendations"].append(
            f"Increase noise bound for better security (currently {noise_ratio:.6f})"
        )
    
    # Overall assessment
    if len(security_report["recommendations"]) == 0:
        security_report["overall_security"] = "good"
    elif len(security_report["recommendations"]) <= 2:
        security_report["overall_security"] = "acceptable"
    else:
        security_report["overall_security"] = "needs_improvement"
    
    return security_report