#!/usr/bin/env python3
"""
SimplePIR Basic Usage Example
Demonstrates fundamental PIR operations
"""

import sys
import time
sys.path.append('..')

from simple_pir import SimplePIRProtocol, SimplePIRConfig, SecurityLevel


def basic_example():
    """Basic SimplePIR usage example"""
    print("=== SimplePIR Basic Usage Example ===\n")
    
    # Create a sample database
    database = [f"Document_{i:03d}: This is secret document number {i}" 
               for i in range(50)]
    print(f"Created database with {len(database)} documents")
    
    # Initialize SimplePIR with medium security
    config = SimplePIRConfig(SecurityLevel.MEDIUM)
    pir_protocol = SimplePIRProtocol(database, config)
    
    print(f"SimplePIR initialized with {config.security_level.value} security")
    print(f"Matrix dimensions: {pir_protocol.server.rows}x{pir_protocol.server.cols}\n")
    
    # Retrieve a specific document privately
    target_index = 23
    print(f"Privately retrieving document at index {target_index}...")
    
    start_time = time.time()
    result = pir_protocol.retrieve_item(target_index)
    end_time = time.time()
    
    if result["retrieval_successful"]:
        print(f"[SUCCESS] Retrieved: {result['item_data']}")
        print(f"[TIME] Total time: {end_time - start_time:.4f}s")
        print(f"[STATS] Performance breakdown:")
        breakdown = result["performance_breakdown"]
        print(f"   - Query generation: {breakdown['query_generation_time']:.4f}s")
        print(f"   - Server processing: {breakdown['server_processing_time']:.4f}s")
        print(f"   - Response decryption: {breakdown['decryption_time']:.4f}s")
    else:
        print(f"[FAILED] Retrieval failed: {result.get('error', 'Unknown error')}")
    
    print()


def batch_example():
    """Batch retrieval example"""
    print("=== Batch Retrieval Example ===\n")
    
    # Create a larger database
    database = [f"Record_{i:04d}" for i in range(200)]
    
    # Use high security for this example
    config = SimplePIRConfig(SecurityLevel.HIGH)
    config.enable_batching = True
    config.batch_size = 16
    
    pir_protocol = SimplePIRProtocol(database, config)
    print(f"Created PIR protocol for {len(database)} records with batching enabled")
    
    # Retrieve multiple items
    items_to_retrieve = [5, 23, 67, 89, 134, 156, 178, 199]
    print(f"Retrieving {len(items_to_retrieve)} items: {items_to_retrieve}")
    
    start_time = time.time()
    batch_results = pir_protocol.batch_retrieve_items(items_to_retrieve)
    batch_time = time.time() - start_time
    
    print(f"[SUCCESS] Batch retrieval completed in {batch_time:.4f}s")
    print(f"[SPEED] Average time per item: {batch_time/len(items_to_retrieve):.4f}s")
    
    print("\nRetrieved items:")
    for result in batch_results:
        if result["retrieval_successful"]:
            print(f"  - Index {result['item_index']}: {result['item_data']}")
        else:
            print(f"  - Index {result['item_index']}: FAILED")
    
    print()


def security_levels_example():
    """Compare different security levels"""
    print("=== Security Levels Comparison ===\n")
    
    database = [f"Item {i}" for i in range(100)]
    security_levels = [SecurityLevel.LOW, SecurityLevel.MEDIUM, 
                      SecurityLevel.HIGH, SecurityLevel.ULTRA]
    
    print("Security Level | Setup Time | Query Time | Communication")
    print("-" * 55)
    
    for level in security_levels:
        config = SimplePIRConfig(level)
        protocol = SimplePIRProtocol(database, config)
        
        # Measure setup time (already done during initialization)
        setup_time = protocol.server.performance_stats["setup_time"]
        
        # Measure query time
        start_time = time.time()
        result = protocol.retrieve_item(42)
        query_time = time.time() - start_time
        
        # Estimate communication cost
        comm_cost = config.estimate_communication_cost(len(database))
        comm_kb = comm_cost["total_communication_bytes"] / 1024
        
        print(f"{level.value:12s}   | {setup_time:8.4f}s | {query_time:8.4f}s | {comm_kb:8.1f} KB")
    
    print()


def performance_benchmark():
    """Performance benchmarking example"""
    print("=== Performance Benchmark ===\n")
    
    # Create database
    database = [f"Benchmark_Document_{i:04d}" for i in range(500)]
    
    # Use medium security for benchmark
    config = SimplePIRConfig(SecurityLevel.MEDIUM)
    config.enable_preprocessing = True
    
    pir_protocol = SimplePIRProtocol(database, config)
    
    print(f"Running benchmark with {len(database)} documents...")
    
    # Run benchmark
    benchmark_results = pir_protocol.benchmark_performance(num_retrievals=20)
    
    summary = benchmark_results["benchmark_summary"]
    print(f"\n[STATS] Benchmark Results:")
    print(f"   Total retrievals: {summary['total_retrievals']}")
    print(f"   Success rate: {summary['success_rate']*100:.1f}%")
    print(f"   Average retrieval time: {summary['average_retrieval_time']:.4f}s")
    print(f"   Min retrieval time: {summary['min_retrieval_time']:.4f}s")
    print(f"   Max retrieval time: {summary['max_retrieval_time']:.4f}s")
    print(f"   Throughput: {summary['throughput_items_per_second']:.2f} items/s")
    
    # Show protocol statistics
    protocol_stats = benchmark_results["protocol_stats"]
    server_stats = protocol_stats["server_stats"]
    client_stats = protocol_stats["client_stats"]
    
    print(f"\n[SERVER] Statistics:")
    print(f"   Setup time: {server_stats['setup_time']:.4f}s")
    print(f"   Queries processed: {server_stats['queries_processed']}")
    print(f"   Average query processing: {server_stats['average_query_time']:.4f}s")
    print(f"   Memory usage estimate: {server_stats['memory_usage_estimate']}")
    
    print(f"\n[CLIENT] Statistics:")
    print(f"   Queries generated: {client_stats['queries_generated']}")
    print(f"   Average generation time: {client_stats['average_generation_time']:.4f}s")
    print(f"   Average decryption time: {client_stats['average_decryption_time']:.4f}s")
    
    print()


def main():
    """Run all examples"""
    try:
        basic_example()
        batch_example()
        security_levels_example()
        performance_benchmark()
        
        print("[SUCCESS] All examples completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()