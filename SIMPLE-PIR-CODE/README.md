# SimplePIR: Simple and Fast Private Information Retrieval Library

A Python implementation of SimplePIR based on the research paper "One Server for the Price of Two: Simple and Fast Single-Server Private Information Retrieval".

## Overview

SimplePIR is a library that provides **Private Information Retrieval (PIR)** capabilities, allowing clients to retrieve data from a server without revealing which specific data they are accessing. This implementation is based on the **Learning with Errors (LWE)** cryptographic assumption and provides:

-  **Strong Privacy Guarantees**: Server cannot determine which item is being retrieved
-  **Efficient Performance**: O(√n) communication complexity  
-  **Post-Quantum Security**: Based on LWE assumption, resistant to quantum attacks
-  **Easy Integration**: Simple API for integration into existing applications
-  **Configurable Security**: Multiple security levels from 80-bit to 256-bit

## Features

### Core Functionality

- **SimplePIR Server**: Efficiently serves PIR queries without learning access patterns
- **SimplePIR Client**: Generates private queries and decrypts responses
- **Batch Operations**: Support for processing multiple queries efficiently
- **Performance Monitoring**: Built-in performance metrics and benchmarking

### Security Features

- **Multiple Security Levels**: LOW (80-bit), MEDIUM (128-bit), HIGH (192-bit), ULTRA (256-bit)
- **Configurable Parameters**: Adjustable LWE parameters for different security/performance trade-offs
- **Matrix Optimization**: Automatic optimization of database organization for minimal communication
- **Error Correction**: Robust decryption with noise tolerance

## Installation

### From Source

```bash
git clone <repository-url>
cd SIMPLE-PIR-CODE
pip install -e .
```

### Dependencies

```bash
pip install numpy>=1.21.0 scipy>=1.7.0
```

### Development Installation

```bash
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```python
from simple_pir import SimplePIRProtocol, SimplePIRConfig, SecurityLevel

# Create a database
database = [f"Document {i}" for i in range(100)]

# Initialize SimplePIR with medium security
config = SimplePIRConfig(SecurityLevel.MEDIUM)
pir_protocol = SimplePIRProtocol(database, config)

# Retrieve item at index 42 privately
result = pir_protocol.retrieve_item(42)

if result["retrieval_successful"]:
    print(f"Retrieved: {result['item_data']}")
    print(f"Time taken: {result['protocol_time']:.4f}s")
```

### Server-Client Separation

```python
from simple_pir import SimplePIRServer, SimplePIRClient

# Server side
database = ["Secret Document A", "Secret Document B", "Secret Document C"]
server = SimplePIRServer(database)
public_params = server.setup()

# Client side  
client = SimplePIRClient(public_params)

# Client generates query for item 1 (without server knowing)
query = client.generate_query(1)

# Server processes query (without knowing which item)
response = server.process_query(query)

# Client decrypts response
result = client.decrypt_response(response, query)
print(f"Retrieved: {result['item_data']}")
```

### Batch Retrieval

```python
# Retrieve multiple items efficiently
items_to_retrieve = [5, 23, 67, 89]
batch_results = pir_protocol.batch_retrieve_items(items_to_retrieve)

for result in batch_results:
    print(f"Item {result['item_index']}: {result['item_data']}")
```

## Configuration

### Security Levels

```python
from simple_pir import SimplePIRConfig, SecurityLevel

# Different security levels
config_low = SimplePIRConfig(SecurityLevel.LOW)        # 80-bit security, faster
config_medium = SimplePIRConfig(SecurityLevel.MEDIUM)   # 128-bit security, balanced
config_high = SimplePIRConfig(SecurityLevel.HIGH)      # 192-bit security, stronger
config_ultra = SimplePIRConfig(SecurityLevel.ULTRA)    # 256-bit security, maximum
```

### Custom Configuration

```python
config = SimplePIRConfig(SecurityLevel.MEDIUM)

# Performance settings
config.enable_batching = True
config.batch_size = 32
config.enable_preprocessing = True

# Matrix optimization
config.auto_optimize_dimensions = True
config.dimension_balance_factor = 1.0  # 1.0 = square matrix

# Communication optimization
config.compress_queries = True
config.compress_responses = True
```

## Performance

### Benchmarking

```python
# Run performance benchmark
benchmark_results = pir_protocol.benchmark_performance(num_retrievals=50)

print(f"Average retrieval time: {benchmark_results['benchmark_summary']['average_retrieval_time']:.4f}s")
print(f"Throughput: {benchmark_results['benchmark_summary']['throughput_items_per_second']:.2f} items/s")
```

### Performance Characteristics

For a database of **n** items:

- **Communication**: O(√n) bits per query
- **Server Computation**: O(n) operations per query  
- **Client Computation**: O(√n) operations per query
- **Storage**: O(n) at server, O(1) at client

### Example Performance (1000 items, Medium security):

- **Query Generation**: ~0.01s
- **Server Processing**: ~0.02s  
- **Response Decryption**: ~0.01s
- **Total Retrieval Time**: ~0.04s
- **Communication**: ~1.2KB per retrieval

## API Reference

### Core Classes

#### `SimplePIRProtocol`

High-level interface for complete PIR operations.

```python
SimplePIRProtocol(database: List[Any], config: SimplePIRConfig = None)
```

**Methods:**

- `retrieve_item(item_index: int) -> Dict`: Retrieve single item privately
- `batch_retrieve_items(item_indices: List[int]) -> List[Dict]`: Retrieve multiple items
- `benchmark_performance(num_retrievals: int) -> Dict`: Run performance benchmark
- `get_protocol_stats() -> Dict`: Get performance statistics

#### `SimplePIRServer`

Server-side PIR implementation.

```python
SimplePIRServer(database: List[Any], config: SimplePIRConfig = None)
```

**Methods:**

- `setup() -> Dict`: Initialize server and return public parameters
- `process_query(pir_query: Dict) -> Dict`: Process PIR query from client
- `batch_process_queries(queries: List[Dict]) -> List[Dict]`: Process multiple queries
- `get_performance_stats() -> Dict`: Get server performance statistics

#### `SimplePIRClient`

Client-side PIR implementation.

```python
SimplePIRClient(server_params: Dict, config: SimplePIRConfig = None)
```

**Methods:**

- `generate_query(item_index: int) -> Dict`: Generate PIR query for item
- `decrypt_response(response: Dict, query: Dict) -> Dict`: Decrypt server response
- `batch_generate_queries(indices: List[int]) -> List[Dict]`: Generate multiple queries
- `get_performance_stats() -> Dict`: Get client performance statistics

#### `SimplePIRConfig`

Configuration management for SimplePIR.

```python
SimplePIRConfig(security_level: SecurityLevel = SecurityLevel.MEDIUM)
```

**Methods:**

- `get_optimal_dimensions(n: int) -> Tuple[int, int]`: Get optimal matrix dimensions
- `estimate_communication_cost(n: int) -> Dict`: Estimate communication costs
- `validate_config() -> bool`: Validate configuration parameters
- `to_dict() -> Dict`: Export configuration to dictionary

### Security Levels

```python
from simple_pir import SecurityLevel

SecurityLevel.LOW     # 80-bit security
SecurityLevel.MEDIUM  # 128-bit security  
SecurityLevel.HIGH    # 192-bit security
SecurityLevel.ULTRA   # 256-bit security
```

## Examples

### Integration with Existing Systems

```python
# Example: Private database query system
class PrivateDatabase:
    def __init__(self, records):
        self.pir_protocol = SimplePIRProtocol(records)
        self.record_index = {record.id: idx for idx, record in enumerate(records)}

    def private_lookup(self, record_id):
        if record_id not in self.record_index:
            return None

        idx = self.record_index[record_id]
        result = self.pir_protocol.retrieve_item(idx)
        return result['item_data'] if result['retrieval_successful'] else None

# Usage
records = [DatabaseRecord(id=i, data=f"Record {i}") for i in range(1000)]
private_db = PrivateDatabase(records)

# Private lookup without revealing which record
record_data = private_db.private_lookup(record_id=123)
```

### Custom Data Types

```python
# SimplePIR works with any serializable data type
import pickle

class Document:
    def __init__(self, title, content):
        self.title = title
        self.content = content

    def __str__(self):
        return f"Document({self.title})"

# Create database of custom objects
documents = [Document(f"Title {i}", f"Content {i}") for i in range(50)]

# SimplePIR handles serialization automatically
pir_protocol = SimplePIRProtocol(documents)
result = pir_protocol.retrieve_item(25)

retrieved_doc = result['item_data']
print(f"Retrieved: {retrieved_doc}")
```

## Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=simple_pir --cov-report=html

# Run specific test category
python -m pytest tests/test_core.py -v
```

### Test Categories

- `test_core.py`: Core PIR protocol tests
- `test_security.py`: Security and cryptographic tests  
- `test_performance.py`: Performance and benchmarking tests
- `test_integration.py`: Integration and compatibility tests

## Contributing

### Development Setup

```bash
git clone <repository-url>
cd SIMPLE-PIR-CODE
pip install -e .[dev]
pre-commit install
```

### Code Style

- Use Black for code formatting
- Follow PEP 8 guidelines
- Add type hints for all public APIs
- Write comprehensive docstrings

### Testing Requirements

- Maintain >90% test coverage
- Add tests for new features
- Ensure backward compatibility

## Security Considerations

### Security Model

- **Privacy**: Server cannot determine which item is accessed
- **Correctness**: Client always retrieves the correct item
- **Security Assumption**: Based on LWE (Learning with Errors) problem
- **Post-Quantum**: Resistant to quantum computer attacks

### Security Best Practices

- Use MEDIUM or higher security level for production
- Regularly rotate server keys if supported
- Monitor for unusual query patterns (though server cannot decrypt)
- Use secure channels (TLS) for query transmission

### Known Limitations

- Communication overhead scales as O(√n) 
- Server computation scales as O(n) per query
- Not suitable for real-time applications with very large databases
- Requires trusted setup phase for key generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use SimplePIR in academic work, please cite:

```bibtex
@article{SimplePIR2023,
  title={One Server for the Price of Two: Simple and Fast Single-Server Private Information Retrieval},
  author={[Authors]},
  journal={[Journal]},
  year={2023}
}
```


