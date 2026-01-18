"""
SimplePIR Server Implementation
Provides private information retrieval server functionality
"""

import numpy as np
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from ..config.pir_config import SimplePIRConfig
from ..utils.crypto_utils import LWEParameters, generate_noise, compute_inner_product
from ..utils.matrix_utils import optimal_matrix_dimensions, matrix_to_vector, vector_to_matrix


class SimplePIRServer:
    """
    SimplePIR Server implementation for private information retrieval
    
    Implements the server-side algorithms from the SimplePIR paper:
    "One Server for the Price of Two: Simple and Fast Single-Server Private Information Retrieval"
    """
    
    def __init__(self, database: List[Any], config: SimplePIRConfig = None):
        """
        Initialize SimplePIR server
        
        Args:
            database: List of items to serve privately
            config: SimplePIR configuration parameters
        """
        self.database = database
        self.config = config or SimplePIRConfig()
        self.n = len(database)
        
        # Server state
        self.setup_complete = False
        self.preprocessed_data = {}
        self.performance_stats = {
            "setup_time": 0.0,
            "queries_processed": 0,
            "total_query_time": 0.0,
            "average_query_time": 0.0
        }
        
        # Initialize LWE parameters
        self.lwe_params = LWEParameters(self.config.get_lwe_parameters())
        
        print(f"SimplePIR Server initialized with {self.n} database items")
    
    def setup(self) -> Dict[str, Any]:
        """
        Server setup phase - preprocess database for efficient PIR
        
        Returns:
            Public parameters for clients
        """
        print("Setting up SimplePIR server...")
        start_time = time.time()
        
        if not self.config.validate_config():
            raise ValueError("Invalid SimplePIR configuration")
        
        # Calculate optimal matrix dimensions
        self.rows, self.cols = self.config.get_optimal_dimensions(self.n)
        print(f"Using {self.rows}x{self.cols} matrix for {self.n} items")
        
        # Preprocess database into matrix format
        self._preprocess_database()
        
        # Generate server preprocessing if enabled
        if self.config.enable_preprocessing:
            self._generate_preprocessing()
        
        setup_time = time.time() - start_time
        self.performance_stats["setup_time"] = setup_time
        
        # Create public parameters for clients
        public_params = {
            "database_size": self.n,
            "matrix_rows": self.rows,
            "matrix_cols": self.cols,
            "lwe_params": self.lwe_params.to_dict(),
            "config": self.config.to_dict(),
            "server_id": self._generate_server_id(),
            "setup_time": setup_time
        }
        
        self.setup_complete = True
        print(f"Server setup completed in {setup_time:.4f}s")
        
        return public_params
    
    def _preprocess_database(self):
        """Preprocess database into matrix format for efficient PIR operations"""
        # Pad database to fit matrix dimensions
        total_slots = self.rows * self.cols
        padded_database = self.database.copy()
        
        # Add padding items if necessary
        if len(padded_database) < total_slots:
            padding_count = total_slots - len(padded_database)
            padded_database.extend([None] * padding_count)
        
        # Reshape into matrix form
        self.db_matrix = np.array(padded_database).reshape(self.rows, self.cols)
        
        # Create position mapping for original items
        self.item_positions = {}
        for i, item in enumerate(self.database):
            row = i // self.cols
            col = i % self.cols
            self.item_positions[i] = (row, col)
        
        print(f"Database preprocessed into {self.rows}x{self.cols} matrix")
    
    def _generate_preprocessing(self):
        """Generate server-side preprocessing for improved performance"""
        # 修复：Windows兼容性 - 避免NumPy int32溢出
        # 限制模数范围以防止溢出，使用int64确保安全
        safe_modulus = min(self.lwe_params.modulus, 2**31 - 1)
        
        print(f"Using safe modulus: {safe_modulus} (original: {self.lwe_params.modulus})")
        
        # Generate random preprocessing matrices (Windows兼容版本)
        self.preprocess_matrix_A = np.random.randint(
            0, safe_modulus, 
            size=(self.rows, self.lwe_params.dimension),
            dtype=np.int64
        )
        self.preprocess_matrix_B = np.random.randint(
            0, safe_modulus,
            size=(self.cols, self.lwe_params.dimension),
            dtype=np.int64
        )
        
        print("Server preprocessing completed (Windows compatible)")
    
    def _generate_server_id(self) -> str:
        """Generate unique server identifier"""
        content = f"{self.n}_{self.rows}_{self.cols}_{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def process_query(self, pir_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process PIR query from client
        
        Args:
            pir_query: PIR query containing encrypted vectors
            
        Returns:
            PIR response containing encrypted result
        """
        if not self.setup_complete:
            raise ValueError("Server setup must be completed before processing queries")
        
        start_time = time.time()
        
        # Extract and deserialize LWE query components
        query_vector_A_raw = pir_query["query_vector_A"]
        query_vector_B_raw = pir_query["query_vector_B"]
        query_id = pir_query.get("query_id", "unknown")
        
        # 反序列化LWE密文结构
        def deserialize_lwe_vector(serialized_vector):
            lwe_vector = []
            for ciphertext_dict in serialized_vector:
                lwe_vector.append({
                    'a': np.array(ciphertext_dict['a'], dtype=np.int64),
                    'b': ciphertext_dict['b']
                })
            return lwe_vector
        
        query_vector_A = deserialize_lwe_vector(query_vector_A_raw)
        query_vector_B = deserialize_lwe_vector(query_vector_B_raw)
        
        print(f"Processing PIR query {query_id}")
        
        # Validate query dimensions
        if len(query_vector_A) != self.rows or len(query_vector_B) != self.cols:
            raise ValueError(f"Invalid query dimensions: expected ({self.rows}, {self.cols}), "
                           f"got ({len(query_vector_A)}, {len(query_vector_B)})")
        
        # Compute PIR response vectors
        response_A = self._compute_response_vector_A(query_vector_A)
        response_B = self._compute_response_vector_B(query_vector_B)
        
        query_time = time.time() - start_time
        
        # Update performance statistics
        self._update_performance_stats(query_time)
        
        # 序列化LWE响应密文以便传输
        def serialize_lwe_response_vector(lwe_response_vector):
            serialized = []
            for lwe_ciphertext in lwe_response_vector:
                serialized.append({
                    'a': lwe_ciphertext['a'].tolist(),
                    'b': int(lwe_ciphertext['b'])
                })
            return serialized
        
        # Create response
        pir_response = {
            "response_vector_A": serialize_lwe_response_vector(response_A),
            "response_vector_B": serialize_lwe_response_vector(response_B),
            "query_id": query_id,
            "server_id": self._generate_server_id(),
            "processing_time": query_time,
            "timestamp": time.time()
        }
        
        print(f"PIR query {query_id} processed in {query_time:.4f}s")
        
        return pir_response
    
    def _compute_response_vector_A(self, query_vector_A: np.ndarray) -> np.ndarray:
        """
        Compute response vector A for PIR query
        
        Args:
            query_vector_A: Query vector for rows
            
        Returns:
            Response vector A
        """
        response_A = np.zeros(self.cols, dtype=object)
        
        for col in range(self.cols):
            # Extract column from database matrix
            column_data = self.db_matrix[:, col]
            
            # Compute homomorphic inner product
            response_A[col] = self._homomorphic_inner_product(
                query_vector_A, column_data
            )
        
        return response_A
    
    def _compute_response_vector_B(self, query_vector_B: np.ndarray) -> np.ndarray:
        """
        Compute response vector B for PIR query
        
        Args:
            query_vector_B: Query vector for columns
            
        Returns:
            Response vector B
        """
        response_B = np.zeros(self.rows, dtype=object)
        
        for row in range(self.rows):
            # Extract row from database matrix
            row_data = self.db_matrix[row, :]
            
            # Compute homomorphic inner product
            response_B[row] = self._homomorphic_inner_product(
                query_vector_B, row_data
            )
        
        return response_B
    
    def _homomorphic_inner_product(self, query_vector: list, 
                                 data_vector: np.ndarray) -> dict:
        """
        Compute homomorphic inner product using vectorized operations (OPTIMIZED)
        
        Args:
            query_vector: List of LWE encrypted query elements
            data_vector: Database data vector (plaintext)
            
        Returns:
            LWE encrypted result of homomorphic inner product
        """
        # 预先提取所有密文组件进行向量化处理
        query_a_matrix = []  # shape: [vector_length, lwe_dimension]
        query_b_vector = []  # shape: [vector_length]
        data_scalars = []    # shape: [vector_length]
        
        for lwe_ciphertext, data_item in zip(query_vector, data_vector):
            if data_item is not None:
                # 数据编码向量化预处理
                if isinstance(data_item, str):
                    data_scalar = hash(data_item) % 1000  # 限制标量范围
                else:
                    data_scalar = int(data_item) % 1000
                
                query_a_matrix.append(lwe_ciphertext['a'])
                query_b_vector.append(lwe_ciphertext['b'])
                data_scalars.append(data_scalar)
        
        if not query_a_matrix:  # 处理空向量情况
            return {'a': np.zeros(self.lwe_params.dimension, dtype=np.int64), 'b': 0}
        
        # 转换为NumPy数组进行向量化计算
        A_matrix = np.array(query_a_matrix, dtype=np.int64)      # [N, d]
        b_vector = np.array(query_b_vector, dtype=np.int64)      # [N]
        scalar_vector = np.array(data_scalars, dtype=np.int64)   # [N]
        
        # 向量化同态乘法: 所有密文同时乘以对应标量
        # Enc(m) * scalar = (a*scalar, b*scalar)
        scaled_A = (A_matrix * scalar_vector[:, np.newaxis]) % self.lwe_params.modulus  # [N, d]
        scaled_b = (b_vector * scalar_vector) % self.lwe_params.modulus                # [N]
        
        # 向量化同态加法: 所有密文求和
        # Sum(Enc(m_i)) = (sum(a_i), sum(b_i))
        result_a = np.sum(scaled_A, axis=0) % self.lwe_params.modulus  # [d]
        result_b = np.sum(scaled_b) % self.lwe_params.modulus          # scalar
        
        return {'a': result_a, 'b': result_b}
    
    def _update_performance_stats(self, query_time: float):
        """Update server performance statistics"""
        self.performance_stats["queries_processed"] += 1
        self.performance_stats["total_query_time"] += query_time
        
        count = self.performance_stats["queries_processed"]
        total_time = self.performance_stats["total_query_time"]
        self.performance_stats["average_query_time"] = total_time / count
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get server performance statistics
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "setup_time": self.performance_stats["setup_time"],
            "queries_processed": self.performance_stats["queries_processed"],
            "average_query_time": self.performance_stats["average_query_time"],
            "total_query_time": self.performance_stats["total_query_time"],
            "database_size": self.n,
            "matrix_dimensions": f"{self.rows}x{self.cols}",
            "memory_usage_estimate": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> str:
        """Estimate server memory usage"""
        # Rough estimate of memory usage
        db_size = len(str(self.database)) 
        matrix_size = self.rows * self.cols * 8  # 8 bytes per pointer
        preprocessing_size = 0
        
        if self.config.enable_preprocessing:
            preprocessing_size = (
                self.preprocess_matrix_A.nbytes + 
                self.preprocess_matrix_B.nbytes
            )
        
        total_bytes = db_size + matrix_size + preprocessing_size
        
        if total_bytes < 1024:
            return f"{total_bytes} bytes"
        elif total_bytes < 1024**2:
            return f"{total_bytes/1024:.1f} KB"
        elif total_bytes < 1024**3:
            return f"{total_bytes/(1024**2):.1f} MB"
        else:
            return f"{total_bytes/(1024**3):.1f} GB"
    
    def batch_process_queries(self, pir_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple PIR queries in batch
        
        Args:
            pir_queries: List of PIR queries
            
        Returns:
            List of PIR responses
        """
        if not self.config.enable_batching:
            # Process queries individually
            return [self.process_query(query) for query in pir_queries]
        
        print(f"Processing batch of {len(pir_queries)} PIR queries")
        batch_start_time = time.time()
        
        responses = []
        for query in pir_queries:
            response = self.process_query(query)
            responses.append(response)
        
        batch_time = time.time() - batch_start_time
        print(f"Batch processing completed in {batch_time:.4f}s")
        print(f"Average time per query: {batch_time/len(pir_queries):.4f}s")
        
        return responses
    
    def shutdown(self):
        """Cleanup server resources"""
        print("Shutting down SimplePIR server...")
        
        # Clear sensitive data
        if hasattr(self, 'db_matrix'):
            del self.db_matrix
        if hasattr(self, 'preprocess_matrix_A'):
            del self.preprocess_matrix_A
        if hasattr(self, 'preprocess_matrix_B'):
            del self.preprocess_matrix_B
        
        self.setup_complete = False
        print("SimplePIR server shutdown complete")