"""
Matrix utilities for SimplePIR
Database organization and matrix operations for efficient PIR
"""

import math
import numpy as np
from typing import Tuple, List, Any, Optional


def optimal_matrix_dimensions(n: int, balance_factor: float = 1.0,
                            min_size: int = 16, max_size: int = 2048) -> Tuple[int, int]:
    """
    Calculate optimal matrix dimensions for n items
    
    Args:
        n: Number of items to organize
        balance_factor: Balance between rows and columns (1.0 = square)
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        
    Returns:
        Tuple of (rows, cols) for optimal dimensions
    """
    if n <= 0:
        raise ValueError("Number of items must be positive")
    
    if n == 1:
        return (1, 1)
    
    # Calculate base square root
    sqrt_n = math.ceil(math.sqrt(n))
    
    if balance_factor == 1.0:
        # Perfect square matrix
        rows = cols = sqrt_n
    else:
        # Adjust for balance factor
        if balance_factor > 1.0:
            # More rows than columns
            rows = min(max_size, max(min_size, int(sqrt_n * balance_factor)))
            cols = math.ceil(n / rows)
        else:
            # More columns than rows
            cols = min(max_size, max(min_size, int(sqrt_n / balance_factor)))
            rows = math.ceil(n / cols)
    
    # Ensure dimensions are within bounds
    rows = max(min_size, min(max_size, rows))
    cols = max(1, math.ceil(n / rows))
    
    # Verify we can fit all items
    if rows * cols < n:
        # Adjust to ensure we have enough space
        if rows < max_size:
            rows = math.ceil(n / cols)
        else:
            cols = math.ceil(n / rows)
    
    return (rows, cols)


def matrix_to_vector(matrix: np.ndarray, order: str = 'row') -> np.ndarray:
    """
    Convert matrix to vector
    
    Args:
        matrix: Input matrix
        order: Ordering ('row' or 'col')
        
    Returns:
        Flattened vector
    """
    if order == 'row':
        return matrix.flatten('C')  # Row-major order
    elif order == 'col':
        return matrix.flatten('F')  # Column-major order
    else:
        raise ValueError("Order must be 'row' or 'col'")


def vector_to_matrix(vector: np.ndarray, rows: int, cols: int,
                    order: str = 'row') -> np.ndarray:
    """
    Convert vector to matrix
    
    Args:
        vector: Input vector
        rows: Number of rows
        cols: Number of columns
        order: Ordering ('row' or 'col')
        
    Returns:
        Reshaped matrix
    """
    if len(vector) != rows * cols:
        raise ValueError(f"Vector length {len(vector)} doesn't match matrix size {rows}x{cols}")
    
    if order == 'row':
        return vector.reshape(rows, cols, order='C')
    elif order == 'col':
        return vector.reshape(rows, cols, order='F')
    else:
        raise ValueError("Order must be 'row' or 'col'")


def pad_database(database: List[Any], target_size: int, 
                padding_value: Any = None) -> List[Any]:
    """
    Pad database to target size
    
    Args:
        database: Original database
        target_size: Target size after padding
        padding_value: Value to use for padding
        
    Returns:
        Padded database
    """
    if len(database) >= target_size:
        return database[:target_size]
    
    padding_needed = target_size - len(database)
    padded_database = database.copy()
    padded_database.extend([padding_value] * padding_needed)
    
    return padded_database


def organize_database_matrix(database: List[Any], rows: int, cols: int) -> np.ndarray:
    """
    Organize database into matrix format
    
    Args:
        database: Database items
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        Database organized as matrix
    """
    target_size = rows * cols
    padded_db = pad_database(database, target_size)
    
    db_matrix = np.array(padded_db).reshape(rows, cols)
    return db_matrix


def create_position_map(database: List[Any], rows: int, cols: int) -> dict:
    """
    Create mapping from item indices to matrix positions
    
    Args:
        database: Original database
        rows: Number of matrix rows
        cols: Number of matrix columns
        
    Returns:
        Dictionary mapping item index to (row, col)
    """
    position_map = {}
    
    for i, item in enumerate(database):
        row = i // cols
        col = i % cols
        position_map[i] = (row, col)
    
    return position_map


def calculate_communication_overhead(n: int, rows: int, cols: int,
                                   element_size_bits: int = 32) -> dict:
    """
    Calculate communication overhead for PIR with given matrix dimensions
    
    Args:
        n: Number of database items
        rows: Number of matrix rows
        cols: Number of matrix columns
        element_size_bits: Size of each element in bits
        
    Returns:
        Dictionary with communication analysis
    """
    # Query consists of two vectors
    query_bits = (rows + cols) * element_size_bits
    
    # Response consists of two vectors
    response_bits = (rows + cols) * element_size_bits
    
    # Total communication
    total_bits = query_bits + response_bits
    
    # Compare to naive approach (downloading entire database)
    naive_bits = n * element_size_bits
    
    # Calculate efficiency metrics
    communication_ratio = total_bits / naive_bits if naive_bits > 0 else float('inf')
    sqrt_n_theoretical = 2 * math.sqrt(n) * element_size_bits
    optimality_ratio = total_bits / sqrt_n_theoretical if sqrt_n_theoretical > 0 else float('inf')
    
    return {
        "query_bits": query_bits,
        "response_bits": response_bits,
        "total_communication_bits": total_bits,
        "naive_download_bits": naive_bits,
        "communication_ratio": communication_ratio,
        "sqrt_n_theoretical_bits": sqrt_n_theoretical,
        "optimality_ratio": optimality_ratio,
        "matrix_efficiency": f"{rows}x{cols} for {n} items"
    }


def find_best_dimensions(n: int, max_communication: Optional[int] = None,
                        element_size_bits: int = 32) -> Tuple[int, int, dict]:
    """
    Find best matrix dimensions for given constraints
    
    Args:
        n: Number of database items
        max_communication: Maximum allowed communication in bits
        element_size_bits: Size of each element in bits
        
    Returns:
        Tuple of (best_rows, best_cols, analysis)
    """
    best_dims = None
    best_communication = float('inf')
    best_analysis = None
    
    # Try different balance factors
    balance_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    for balance in balance_factors:
        rows, cols = optimal_matrix_dimensions(n, balance)
        analysis = calculate_communication_overhead(n, rows, cols, element_size_bits)
        
        total_comm = analysis["total_communication_bits"]
        
        # Check if it meets constraints
        if max_communication is not None and total_comm > max_communication:
            continue
        
        # Check if it's better than current best
        if total_comm < best_communication:
            best_communication = total_comm
            best_dims = (rows, cols)
            best_analysis = analysis
            best_analysis["balance_factor"] = balance
    
    if best_dims is None:
        # Fall back to square dimensions
        rows, cols = optimal_matrix_dimensions(n)
        best_analysis = calculate_communication_overhead(n, rows, cols, element_size_bits)
        best_analysis["balance_factor"] = 1.0
        best_dims = (rows, cols)
    
    return best_dims[0], best_dims[1], best_analysis


def validate_matrix_dimensions(n: int, rows: int, cols: int) -> dict:
    """
    Validate matrix dimensions for database size
    
    Args:
        n: Number of database items
        rows: Proposed number of rows
        cols: Proposed number of columns
        
    Returns:
        Validation report
    """
    validation = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    # Check if matrix can hold all items
    matrix_capacity = rows * cols
    if matrix_capacity < n:
        validation["valid"] = False
        validation["errors"].append(
            f"Matrix {rows}x{cols} too small for {n} items (capacity: {matrix_capacity})"
        )
    
    # Check for excessive waste
    waste_ratio = (matrix_capacity - n) / n if n > 0 else 0
    if waste_ratio > 1.0:  # More than 100% waste
        validation["warnings"].append(
            f"Matrix wastes {waste_ratio:.1%} space ({matrix_capacity - n} empty slots)"
        )
        validation["recommendations"].append("Consider reducing matrix dimensions")
    
    # Check for extreme aspect ratios
    aspect_ratio = max(rows, cols) / min(rows, cols)
    if aspect_ratio > 10:
        validation["warnings"].append(
            f"Extreme aspect ratio {aspect_ratio:.1f}:1 may increase communication cost"
        )
        validation["recommendations"].append("Consider more balanced dimensions")
    
    # Check minimum dimensions
    if rows < 2 or cols < 2:
        validation["warnings"].append("Very small dimensions may not provide good security")
    
    return validation


def benchmark_matrix_organizations(n: int, max_candidates: int = 10) -> List[dict]:
    """
    Benchmark different matrix organizations
    
    Args:
        n: Number of database items
        max_candidates: Maximum number of organizations to test
        
    Returns:
        List of benchmark results sorted by efficiency
    """
    candidates = []
    
    # Generate candidate dimensions
    sqrt_n = int(math.ceil(math.sqrt(n)))
    
    # Test various configurations around sqrt(n)
    test_rows = range(max(1, sqrt_n - 3), sqrt_n + 4)
    
    for rows in test_rows:
        cols = math.ceil(n / rows)
        
        # Skip if too unbalanced
        if max(rows, cols) / min(rows, cols) > 50:
            continue
        
        analysis = calculate_communication_overhead(n, rows, cols)
        validation = validate_matrix_dimensions(n, rows, cols)
        
        candidate = {
            "rows": rows,
            "cols": cols,
            "total_communication": analysis["total_communication_bits"],
            "communication_ratio": analysis["communication_ratio"],
            "optimality_ratio": analysis["optimality_ratio"],
            "validation": validation,
            "matrix_utilization": n / (rows * cols),
            "aspect_ratio": max(rows, cols) / min(rows, cols)
        }
        
        candidates.append(candidate)
        
        if len(candidates) >= max_candidates:
            break
    
    # Sort by total communication cost
    candidates.sort(key=lambda x: x["total_communication"])
    
    return candidates[:max_candidates]