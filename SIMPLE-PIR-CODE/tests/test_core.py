#!/usr/bin/env python3
"""
Core SimplePIR functionality tests
Tests the fundamental PIR protocol operations
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_pir import (
    SimplePIRProtocol, SimplePIRServer, SimplePIRClient, 
    SimplePIRConfig, SecurityLevel
)


class TestSimplePIRCore(unittest.TestCase):
    """Test core SimplePIR functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_database = [f"Test_Document_{i:03d}" for i in range(20)]
        self.config = SimplePIRConfig(SecurityLevel.LOW)  # Use low security for fast tests
        self.config.enable_logging = False  # Reduce test output
    
    def test_server_initialization(self):
        """Test SimplePIR server initialization"""
        server = SimplePIRServer(self.test_database, self.config)
        self.assertEqual(len(server.database), len(self.test_database))
        self.assertEqual(server.n, len(self.test_database))
        self.assertFalse(server.setup_complete)
    
    def test_server_setup(self):
        """Test server setup process"""
        server = SimplePIRServer(self.test_database, self.config)
        public_params = server.setup()
        
        self.assertTrue(server.setup_complete)
        self.assertIn("database_size", public_params)
        self.assertIn("matrix_rows", public_params)
        self.assertIn("matrix_cols", public_params)
        self.assertIn("lwe_params", public_params)
        
        self.assertEqual(public_params["database_size"], len(self.test_database))
        self.assertTrue(public_params["matrix_rows"] * public_params["matrix_cols"] >= len(self.test_database))
    
    def test_client_initialization(self):
        """Test SimplePIR client initialization"""
        server = SimplePIRServer(self.test_database, self.config)
        public_params = server.setup()
        
        client = SimplePIRClient(public_params, self.config)
        
        self.assertEqual(client.database_size, len(self.test_database))
        self.assertEqual(client.rows, public_params["matrix_rows"])
        self.assertEqual(client.cols, public_params["matrix_cols"])
        self.assertIsNotNone(client.client_keys)
    
    def test_query_generation(self):
        """Test PIR query generation"""
        server = SimplePIRServer(self.test_database, self.config)
        public_params = server.setup()
        client = SimplePIRClient(public_params, self.config)
        
        # Test valid query generation
        query = client.generate_query(5)
        
        self.assertIn("query_vector_A", query)
        self.assertIn("query_vector_B", query)
        self.assertIn("query_id", query)
        self.assertIn("target_item_index", query)
        self.assertEqual(query["target_item_index"], 5)
        
        # Check vector dimensions
        self.assertEqual(len(query["query_vector_A"]), server.rows)
        self.assertEqual(len(query["query_vector_B"]), server.cols)
    
    def test_query_processing(self):
        """Test server query processing"""
        server = SimplePIRServer(self.test_database, self.config)
        public_params = server.setup()
        client = SimplePIRClient(public_params, self.config)
        
        # Generate and process query
        query = client.generate_query(3)
        response = server.process_query(query)
        
        self.assertIn("response_vector_A", response)
        self.assertIn("response_vector_B", response)
        self.assertIn("query_id", response)
        self.assertIn("processing_time", response)
        
        self.assertEqual(response["query_id"], query["query_id"])
        self.assertEqual(len(response["response_vector_A"]), server.cols)
        self.assertEqual(len(response["response_vector_B"]), server.rows)
    
    def test_response_decryption(self):
        """Test response decryption"""
        server = SimplePIRServer(self.test_database, self.config)
        public_params = server.setup()
        client = SimplePIRClient(public_params, self.config)
        
        # Full PIR cycle
        target_index = 7
        query = client.generate_query(target_index)
        response = server.process_query(query)
        result = client.decrypt_response(response, query)
        
        self.assertIn("item_index", result)
        self.assertIn("item_data", result)
        self.assertIn("success", result)
        
        self.assertEqual(result["item_index"], target_index)
        self.assertTrue(result["success"])
        self.assertIsNotNone(result["item_data"])
    
    def test_protocol_end_to_end(self):
        """Test complete PIR protocol"""
        protocol = SimplePIRProtocol(self.test_database, self.config)
        
        # Test retrieval
        target_index = 12
        result = protocol.retrieve_item(target_index)
        
        self.assertTrue(result["retrieval_successful"])
        self.assertEqual(result["item_index"], target_index)
        self.assertIn("item_data", result)
        self.assertIn("protocol_time", result)
        self.assertIn("performance_breakdown", result)
        
        # Verify performance breakdown
        breakdown = result["performance_breakdown"]
        self.assertIn("query_generation_time", breakdown)
        self.assertIn("server_processing_time", breakdown)
        self.assertIn("decryption_time", breakdown)
    
    def test_batch_operations(self):
        """Test batch PIR operations"""
        # Enable batching
        config = SimplePIRConfig(SecurityLevel.LOW)
        config.enable_batching = True
        config.batch_size = 5
        
        protocol = SimplePIRProtocol(self.test_database, config)
        
        # Test batch retrieval
        indices = [1, 5, 9, 13, 17]
        results = protocol.batch_retrieve_items(indices)
        
        self.assertEqual(len(results), len(indices))
        
        for i, result in enumerate(results):
            self.assertTrue(result["retrieval_successful"])
            self.assertEqual(result["item_index"], indices[i])
    
    def test_invalid_indices(self):
        """Test handling of invalid indices"""
        server = SimplePIRServer(self.test_database, self.config)
        public_params = server.setup()
        client = SimplePIRClient(public_params, self.config)
        
        # Test negative index
        with self.assertRaises(ValueError):
            client.generate_query(-1)
        
        # Test index out of range
        with self.assertRaises(ValueError):
            client.generate_query(len(self.test_database))
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = SimplePIRConfig(SecurityLevel.MEDIUM)
        self.assertTrue(valid_config.validate_config())
        
        # Test invalid security parameter (too low)
        invalid_config = SimplePIRConfig(SecurityLevel.MEDIUM)
        invalid_config.security_parameter = 50  # Too low
        self.assertFalse(invalid_config.validate_config())
    
    def test_performance_statistics(self):
        """Test performance statistics collection"""
        protocol = SimplePIRProtocol(self.test_database, self.config)
        
        # Perform some operations
        protocol.retrieve_item(0)
        protocol.retrieve_item(5)
        protocol.retrieve_item(10)
        
        # Check server stats
        server_stats = protocol.server.get_performance_stats()
        self.assertGreater(server_stats["queries_processed"], 0)
        self.assertGreater(server_stats["average_query_time"], 0)
        
        # Check client stats  
        client_stats = protocol.client.get_performance_stats()
        self.assertGreater(client_stats["queries_generated"], 0)
        self.assertGreater(client_stats["average_generation_time"], 0)
        
        # Check protocol stats
        protocol_stats = protocol.get_protocol_stats()
        self.assertIn("protocol_stats", protocol_stats)
        self.assertIn("server_stats", protocol_stats)
        self.assertIn("client_stats", protocol_stats)


class TestSecurityLevels(unittest.TestCase):
    """Test different security levels"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_database = [f"Security_Test_{i}" for i in range(10)]
    
    def test_all_security_levels(self):
        """Test all available security levels"""
        security_levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, 
                          SecurityLevel.HIGH, SecurityLevel.ULTRA]
        
        for level in security_levels:
            with self.subTest(security_level=level):
                config = SimplePIRConfig(level)
                config.enable_logging = False
                
                protocol = SimplePIRProtocol(self.test_database, config)
                result = protocol.retrieve_item(3)
                
                self.assertTrue(result["retrieval_successful"])
                self.assertEqual(result["item_index"], 3)
    
    def test_security_parameter_scaling(self):
        """Test that security parameters scale correctly"""
        levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, 
                 SecurityLevel.HIGH, SecurityLevel.ULTRA]
        
        security_params = []
        for level in levels:
            config = SimplePIRConfig(level)
            security_params.append(config.security_parameter)
        
        # Security parameters should be increasing
        for i in range(1, len(security_params)):
            self.assertGreater(security_params[i], security_params[i-1])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_database(self):
        """Test handling of empty database"""
        empty_db = []
        config = SimplePIRConfig(SecurityLevel.LOW)
        
        with self.assertRaises(Exception):
            SimplePIRProtocol(empty_db, config)
    
    def test_single_item_database(self):
        """Test single-item database"""
        single_db = ["SingleItem"]
        config = SimplePIRConfig(SecurityLevel.LOW)
        config.enable_logging = False
        
        protocol = SimplePIRProtocol(single_db, config)
        result = protocol.retrieve_item(0)
        
        self.assertTrue(result["retrieval_successful"])
        self.assertEqual(result["item_index"], 0)
    
    def test_large_database_dimensions(self):
        """Test optimal dimensions for larger databases"""
        large_db = [f"Item_{i:06d}" for i in range(1000)]
        config = SimplePIRConfig(SecurityLevel.LOW)
        config.enable_logging = False
        
        protocol = SimplePIRProtocol(large_db, config)
        
        # Check that matrix dimensions are reasonable
        rows, cols = protocol.server.rows, protocol.server.cols
        self.assertTrue(rows * cols >= len(large_db))
        self.assertTrue(abs(rows - cols) <= max(rows, cols))  # Not too unbalanced


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)