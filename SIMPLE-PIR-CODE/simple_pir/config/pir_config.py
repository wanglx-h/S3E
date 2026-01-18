"""
SimplePIR Configuration
Configuration parameters and security levels for SimplePIR library
"""

from enum import Enum
from typing import Dict, Any
import math


class SecurityLevel(Enum):
    """Security levels for SimplePIR with corresponding LWE parameters"""
    LOW = "low"           # 80-bit security
    MEDIUM = "medium"     # 128-bit security  
    HIGH = "high"         # 192-bit security
    ULTRA = "ultra"       # 256-bit security


class SimplePIRConfig:
    """SimplePIR system configuration parameters"""
    
    # SimplePIR论文标准参数（Windows兼容优化版本）
    # 论文标准：n=1024, q=2^32, σ=6.4
    # 适配Windows：q=2^31-1, 噪声参数按σ*3规则调整
    SECURITY_PARAMS = {
        SecurityLevel.LOW: {
            "security_parameter": 80,
            "dimension": 512,  # 论文标准n，缩减版适配低安全级别
            "modulus": 2**30,  # 1G，平衡安全性与兼容性
            "noise_bound": 16,  # σ≈5.3, 适配低安全级别
            "error_distribution": "gaussian"
        },
        SecurityLevel.MEDIUM: {
            "security_parameter": 128, 
            "dimension": 1024,  # 论文标准LWE维度n=1024
            "modulus": 2**31 - 1,  # 最大int32安全值，接近论文的2^32
            "noise_bound": 19,  # σ≈6.4, 论文标准噪声参数
            "error_distribution": "gaussian"
        },
        SecurityLevel.HIGH: {
            "security_parameter": 192,
            "dimension": 1536,  # 提升LWE维度增强安全性
            "modulus": 2**31 - 1,  # Windows兼容模数
            "noise_bound": 24,  # σ≈8.0, 增强噪声安全性
            "error_distribution": "gaussian"
        },
        SecurityLevel.ULTRA: {
            "security_parameter": 256,
            "dimension": 2048,  # 最高安全级别LWE维度
            "modulus": 2**31 - 1,  # Windows兼容版本
            "noise_bound": 30,  # σ≈10.0, 最高噪声安全性
            "error_distribution": "gaussian"
        }
    }
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """
        Initialize SimplePIR configuration
        
        Args:
            security_level: Desired security level
        """
        self.security_level = security_level
        self._load_security_params()
        
        # Performance parameters
        self.enable_preprocessing = True
        self.enable_batching = True
        self.batch_size = 32
        self.cache_queries = False  # Disable for privacy
        
        # Matrix optimization parameters
        self.auto_optimize_dimensions = True
        self.min_matrix_size = 16
        self.max_matrix_size = 2048
        self.dimension_balance_factor = 1.0  # 1.0 = perfect square
        
        # Communication parameters
        self.compress_queries = True
        self.compress_responses = True
        self.compression_level = 6  # gzip compression level
        
        # Debugging and logging
        self.enable_logging = True
        self.log_performance = True
        self.log_security_warnings = True
        self.debug_mode = False
    
    def _load_security_params(self):
        """Load security parameters based on selected security level"""
        params = self.SECURITY_PARAMS[self.security_level]
        
        self.security_parameter = params["security_parameter"]
        self.dimension = params["dimension"]  # LWE维度n
        self.modulus = params["modulus"]
        self.noise_bound = params["noise_bound"]
        self.error_distribution = params["error_distribution"]
    
    def get_optimal_dimensions(self, n: int) -> tuple:
        """
        Calculate optimal matrix dimensions for n items
        
        Args:
            n: Number of items in database
            
        Returns:
            Tuple of (rows, cols) for optimal matrix dimensions
        """
        if not self.auto_optimize_dimensions:
            sqrt_n = int(math.ceil(math.sqrt(n)))
            return (sqrt_n, sqrt_n)
        
        # Find dimensions that minimize communication cost
        # while maintaining security and performance
        sqrt_n = int(math.ceil(math.sqrt(n)))
        
        # Apply balance factor
        if self.dimension_balance_factor != 1.0:
            rows = max(self.min_matrix_size, 
                      int(sqrt_n / self.dimension_balance_factor))
            cols = int(math.ceil(n / rows))
        else:
            rows = cols = sqrt_n
        
        # Ensure within bounds
        rows = min(max(rows, self.min_matrix_size), self.max_matrix_size)
        cols = max(1, int(math.ceil(n / rows)))
        
        return (rows, cols)
    
    def get_lwe_parameters(self) -> Dict[str, Any]:
        """
        Get LWE parameters for current security level
        
        Returns:
            Dictionary containing LWE parameters
        """
        return {
            "security_parameter": self.security_parameter,
            "dimension": self.dimension,  # 论文标准LWE维度n
            "modulus": self.modulus,
            "noise_bound": self.noise_bound,
            "error_distribution": self.error_distribution
        }
    
    def estimate_communication_cost(self, n: int) -> Dict[str, int]:
        """
        Estimate communication cost for PIR with n items
        
        Args:
            n: Number of items in database
            
        Returns:
            Dictionary with communication cost estimates
        """
        rows, cols = self.get_optimal_dimensions(n)
        
        # Query size: 2 vectors of size (rows, cols) with LWE dimension
        query_size = (rows + cols) * self.dimension * 4  # 4 bytes per int
        
        # Response size: 2 vectors
        response_size = (rows + cols) * 8  # 8 bytes per response element
        
        if self.compress_queries:
            query_size = int(query_size * 0.7)  # Estimate 30% compression
        
        if self.compress_responses:
            response_size = int(response_size * 0.8)  # Estimate 20% compression
        
        return {
            "query_size_bytes": query_size,
            "response_size_bytes": response_size,
            "total_communication_bytes": query_size + response_size,
            "matrix_dimensions": f"{rows}x{cols}"
        }
    
    def validate_config(self) -> bool:
        """
        Validate current configuration
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check security parameters
            if self.security_parameter < 80:
                if self.log_security_warnings:
                    print("WARNING: Security parameter < 80 bits")
                return False
            
            # Check modulus size
            if self.modulus < 2**16:
                if self.log_security_warnings:
                    print("WARNING: Modulus too small for security")
                return False
            
            # Check matrix bounds
            if self.min_matrix_size >= self.max_matrix_size:
                print("ERROR: Invalid matrix size bounds")
                return False
            
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"Config validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "security_level": self.security_level.value,
            "security_parameter": self.security_parameter,
            "dimension": self.dimension,  # LWE维度
            "modulus": self.modulus,
            "noise_bound": self.noise_bound,
            "error_distribution": self.error_distribution,
            "enable_preprocessing": self.enable_preprocessing,
            "enable_batching": self.enable_batching,
            "batch_size": self.batch_size,
            "auto_optimize_dimensions": self.auto_optimize_dimensions,
            "min_matrix_size": self.min_matrix_size,
            "max_matrix_size": self.max_matrix_size,
            "dimension_balance_factor": self.dimension_balance_factor,
            "compress_queries": self.compress_queries,
            "compress_responses": self.compress_responses
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimplePIRConfig':
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            SimplePIRConfig instance
        """
        security_level = SecurityLevel(config_dict.get("security_level", "medium"))
        config = cls(security_level)
        
        # Override with provided values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config