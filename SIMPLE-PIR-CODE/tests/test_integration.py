#!/usr/bin/env python3
"""
Integration tests for SimplePIR library
Tests interaction with external systems and real-world usage patterns
"""

import unittest
import sys
import os
import time
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_pir import SimplePIRProtocol, SimplePIRConfig, SecurityLevel


class TestRealWorldIntegration(unittest.TestCase):
    """Test SimplePIR with real-world usage patterns"""
    
    def setUp(self):
        """Set up test environment with realistic data"""
        # Create realistic database
        self.documents = []
        for i in range(100):
            doc = {
                "id": f"doc_{i:04d}",
                "title": f"Document Title {i}",
                "content": f"This is the content of document {i}. " * 10,
                "metadata": {
                    "created": f"2023-01-{(i%30)+1:02d}",
                    "size": 1000 + i * 50,
                    "category": f"Category_{i%5}"
                }
            }
            self.documents.append(str(doc))  # Convert to string for PIR
        
        self.config = SimplePIRConfig(SecurityLevel.MEDIUM)
        self.config.enable_logging = False
    
    def test_document_database_retrieval(self):
        """Test PIR with document database"""
        protocol = SimplePIRProtocol(self.documents, self.config)
        
        # Retrieve multiple documents
        test_indices = [5, 23, 47, 68, 91]
        
        for idx in test_indices:
            with self.subTest(document_index=idx):
                result = protocol.retrieve_item(idx)
                
                self.assertTrue(result["retrieval_successful"])
                self.assertEqual(result["item_index"], idx)
                
                # Verify we can parse the retrieved document
                retrieved_data = result["item_data"]
                self.assertIn(f"doc_{idx:04d}", str(retrieved_data))
    
    def test_high_frequency_access(self):
        """Test high-frequency access patterns"""
        protocol = SimplePIRProtocol(self.documents, self.config)
        
        # Simulate high-frequency access
        num_queries = 20
        query_results = []
        
        start_time = time.time()
        for _ in range(num_queries):
            idx = random.randint(0, len(self.documents) - 1)
            result = protocol.retrieve_item(idx)
            query_results.append(result)
        
        total_time = time.time() - start_time
        
        # Verify all queries succeeded
        successful_queries = sum(1 for r in query_results if r["retrieval_successful"])
        self.assertEqual(successful_queries, num_queries)
        
        # Check performance is reasonable
        avg_time_per_query = total_time / num_queries
        self.assertLess(avg_time_per_query, 1.0)  # Should be less than 1 second per query
    
    def test_mixed_data_types(self):
        """Test PIR with mixed data types"""
        mixed_database = [
            "Simple string",
            {"key": "value", "number": 42},
            [1, 2, 3, 4, 5],
            "Another string with special chars: !@#$%^&*()",
            {"nested": {"data": "structure"}},
            None,  # Test None values
            "",     # Test empty string
            {"large_text": "Lorem ipsum " * 100}
        ]
        
        # Convert to strings for PIR (in practice, would use proper serialization)
        str_database = [str(item) for item in mixed_database]
        
        protocol = SimplePIRProtocol(str_database, self.config)
        
        for idx in range(len(mixed_database)):
            with self.subTest(item_index=idx):
                result = protocol.retrieve_item(idx)
                self.assertTrue(result["retrieval_successful"])
    
    def test_concurrent_access_simulation(self):
        """Simulate concurrent access patterns (single-threaded simulation)"""
        protocol = SimplePIRProtocol(self.documents, self.config)
        
        # Simulate multiple "clients" accessing different documents
        client_requests = [
            {"client_id": "client_1", "requests": [1, 5, 9, 13]},
            {"client_id": "client_2", "requests": [2, 6, 10, 14]},
            {"client_id": "client_3", "requests": [3, 7, 11, 15]},
        ]
        
        all_results = []
        
        for client in client_requests:
            client_results = []
            for idx in client["requests"]:
                result = protocol.retrieve_item(idx)
                result["client_id"] = client["client_id"]
                client_results.append(result)
                all_results.append(result)
            
            # Verify all requests for this client succeeded
            successful = all(r["retrieval_successful"] for r in client_results)
            self.assertTrue(successful, f"Not all requests succeeded for {client['client_id']}")
        
        # Verify all results are correct
        self.assertEqual(len(all_results), sum(len(c["requests"]) for c in client_requests))
    
    def test_benchmark_realistic_workload(self):
        """Benchmark with realistic workload"""
        protocol = SimplePIRProtocol(self.documents, self.config)
        
        # Run comprehensive benchmark
        benchmark_results = protocol.benchmark_performance(num_retrievals=30, random_items=True)
        
        summary = benchmark_results["benchmark_summary"]
        
        # Verify benchmark results
        self.assertGreater(summary["total_retrievals"], 0)
        self.assertGreaterEqual(summary["success_rate"], 0.9)  # At least 90% success rate
        self.assertGreater(summary["throughput_items_per_second"], 0)
        
        # Performance should be reasonable for medium security
        self.assertLess(summary["average_retrieval_time"], 0.5)  # Less than 500ms average
        
        print(f"\n📊 Realistic Workload Benchmark Results:")
        print(f"   Average retrieval time: {summary['average_retrieval_time']:.4f}s")
        print(f"   Throughput: {summary['throughput_items_per_second']:.2f} items/s")
        print(f"   Success rate: {summary['success_rate']*100:.1f}%")


class TestScalabilityIntegration(unittest.TestCase):
    """Test SimplePIR scalability with different database sizes"""
    
    def test_small_database_scalability(self):
        """Test with small database (10-50 items)"""
        sizes_to_test = [10, 25, 50]
        
        for size in sizes_to_test:
            with self.subTest(database_size=size):
                database = [f"SmallDB_Item_{i:03d}" for i in range(size)]
                config = SimplePIRConfig(SecurityLevel.LOW)
                config.enable_logging = False
                
                protocol = SimplePIRProtocol(database, config)
                
                # Test retrieval
                mid_index = size // 2
                result = protocol.retrieve_item(mid_index)
                
                self.assertTrue(result["retrieval_successful"])
                self.assertEqual(result["item_index"], mid_index)
                
                # Performance should be very fast for small databases
                self.assertLess(result["protocol_time"], 0.1)
    
    def test_medium_database_scalability(self):
        """Test with medium database (100-500 items)"""
        sizes_to_test = [100, 250, 500]
        
        for size in sizes_to_test:
            with self.subTest(database_size=size):
                database = [f"MediumDB_Item_{i:04d}" for i in range(size)]
                config = SimplePIRConfig(SecurityLevel.MEDIUM)
                config.enable_logging = False
                
                protocol = SimplePIRProtocol(database, config)
                
                # Test multiple retrievals
                test_indices = [0, size//4, size//2, 3*size//4, size-1]
                
                for idx in test_indices:
                    result = protocol.retrieve_item(idx)
                    self.assertTrue(result["retrieval_successful"])
                
                # Check matrix dimensions are reasonable
                rows, cols = protocol.server.rows, protocol.server.cols
                expected_sqrt = int(size**0.5) + 1
                self.assertLessEqual(max(rows, cols), expected_sqrt + 5)
    
    def test_large_database_scalability(self):
        """Test with large database (1000+ items)"""
        size = 1000
        database = [f"LargeDB_Item_{i:05d}" for i in range(size)]
        config = SimplePIRConfig(SecurityLevel.LOW)  # Use low security for speed
        config.enable_logging = False
        config.enable_preprocessing = True
        
        protocol = SimplePIRProtocol(database, config)
        
        # Test random retrievals
        random_indices = random.sample(range(size), 10)
        
        total_time = 0
        for idx in random_indices:
            start_time = time.time()
            result = protocol.retrieve_item(idx)
            total_time += time.time() - start_time
            
            self.assertTrue(result["retrieval_successful"])
            self.assertEqual(result["item_index"], idx)
        
        avg_time = total_time / len(random_indices)
        
        # Even for large databases, average time should be reasonable
        self.assertLess(avg_time, 0.5)  # Less than 500ms average
        
        print(f"\n📈 Large Database Performance (n={size}):")
        print(f"   Average retrieval time: {avg_time:.4f}s")
        print(f"   Matrix dimensions: {protocol.server.rows}x{protocol.server.cols}")


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling in integrated scenarios"""
    
    def test_corrupted_query_handling(self):
        """Test handling of corrupted queries"""
        database = [f"Item_{i}" for i in range(20)]
        config = SimplePIRConfig(SecurityLevel.LOW)
        config.enable_logging = False
        
        server = SimplePIRProtocol(database, config).server
        
        # Test malformed query
        malformed_query = {
            "query_vector_A": [1, 2],  # Wrong size
            "query_vector_B": [1, 2, 3],  # Wrong size
            "query_id": "test_query"
        }
        
        with self.assertRaises(Exception):
            server.process_query(malformed_query)
    
    def test_configuration_edge_cases(self):
        """Test edge cases in configuration"""
        database = [f"Config_Test_{i}" for i in range(50)]
        
        # Test with extreme matrix dimensions
        config = SimplePIRConfig(SecurityLevel.LOW)
        config.min_matrix_size = 1
        config.max_matrix_size = 100
        config.dimension_balance_factor = 5.0  # Very unbalanced
        config.enable_logging = False
        
        # Should still work despite unusual configuration
        protocol = SimplePIRProtocol(database, config)
        result = protocol.retrieve_item(25)
        
        self.assertTrue(result["retrieval_successful"])
    
    def test_resource_limits_simulation(self):
        """Simulate resource-constrained environment"""
        database = [f"Resource_Test_{i}" for i in range(200)]
        config = SimplePIRConfig(SecurityLevel.LOW)
        config.enable_preprocessing = False  # Disable to save memory
        config.enable_batching = False       # Disable batching
        config.enable_logging = False
        
        protocol = SimplePIRProtocol(database, config)
        
        # Should work even with minimal features enabled
        result = protocol.retrieve_item(100)
        self.assertTrue(result["retrieval_successful"])
        
        # Performance might be slower but should still be reasonable
        self.assertLess(result["protocol_time"], 2.0)  # Less than 2 seconds


class TestCompatibilityIntegration(unittest.TestCase):
    """Test compatibility with different Python versions and environments"""
    
    def test_data_serialization_compatibility(self):
        """Test that data serialization works consistently"""
        import json
        import pickle
        
        # Test with JSON-serializable data
        json_data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "San Francisco"},
            {"name": "Charlie", "age": 35, "city": "Chicago"}
        ]
        
        json_strings = [json.dumps(item) for item in json_data]
        
        config = SimplePIRConfig(SecurityLevel.LOW)
        config.enable_logging = False
        
        protocol = SimplePIRProtocol(json_strings, config)
        result = protocol.retrieve_item(1)
        
        self.assertTrue(result["retrieval_successful"])
        
        # Should be able to deserialize the result
        retrieved_json = json.loads(result["item_data"])
        self.assertEqual(retrieved_json["name"], "Bob")
    
    def test_unicode_handling(self):
        """Test Unicode string handling"""
        unicode_database = [
            "Hello World",
            "Héllo Wörld",  # Accented characters
            "你好世界",        # Chinese characters
            "🌟⭐🎉",          # Emoji
            "Ω≈ç√∫˜µ≤≥÷",    # Mathematical symbols
        ]
        
        config = SimplePIRConfig(SecurityLevel.LOW)
        config.enable_logging = False
        
        protocol = SimplePIRProtocol(unicode_database, config)
        
        for idx, expected_text in enumerate(unicode_database):
            with self.subTest(text_index=idx):
                result = protocol.retrieve_item(idx)
                self.assertTrue(result["retrieval_successful"])
                # Note: Retrieved data might be encoded differently, 
                # but should contain the original information


if __name__ == "__main__":
    # Run integration tests
    unittest.main(verbosity=2)