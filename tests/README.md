# Unit Tests for Downtown Cab Co

This directory contains comprehensive unit tests for the downtown-cab-co project.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                          # Shared fixtures and configuration
├── inference_api/
│   ├── __init__.py
│   └── test_main.py                    # Tests for inference API endpoints
└── training_api/
    ├── __init__.py
    ├── test_config.py                  # Tests for MLflow configuration
    ├── test_main.py                    # Tests for training API endpoints
    └── data/
        ├── __init__.py
        ├── test_downloader.py          # Tests for data downloading
        ├── test_loader.py              # Tests for data loading
        └── test_processer.py           # Tests for data preprocessing
```

## Running Tests

### Run all tests
```bash
python -m pytest tests/
```

### Run tests with verbose output
```bash
python -m pytest tests/ -v
```

### Run tests for a specific module
```bash
python -m pytest tests/training_api/data/test_loader.py
```

### Run a specific test
```bash
python -m pytest tests/training_api/data/test_loader.py::TestDataLoader::test_load_next_batch_first_batch
```

### Run tests with coverage (requires pytest-cov)
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Coverage

The test suite provides comprehensive coverage for:

### Data Processing Modules
- **test_loader.py** (13 tests): Tests for the DataLoader class
  - File loading and batching
  - Pagination across files
  - Reset and data integrity
  
- **test_processer.py** (19 tests): Tests for the TaxiDataPreprocessor class
  - Data validation and cleaning
  - Outlier detection and removal
  - Feature engineering (temporal, spatial)
  - Data type conversion and missing value handling

- **test_downloader.py** (11 tests): Tests for the DataDownloader class
  - File downloading with streaming
  - Error handling for network failures
  - Multi-year data downloads

### API Modules
- **test_main.py (training)** (12 tests): Tests for training API endpoints
  - Health check endpoint
  - Model training endpoint
  - Model reload functionality

- **test_main.py (inference)** (14 tests): Tests for inference API endpoints
  - Health check endpoint
  - Prediction endpoint
  - Model loading and error handling

### Configuration
- **test_config.py** (4 tests): Tests for MLflow configuration
  - Environment variable handling
  - Default configuration values

## Test Statistics

- **Total tests**: 73
- **Success rate**: 100%
- **Test modules**: 6
- **Average test execution time**: ~2-3 seconds

## Key Features

1. **Mocking**: Extensive use of mocks to isolate units under test
2. **Fixtures**: Shared test fixtures for common test data
3. **Parameterization**: Tests cover various edge cases and scenarios
4. **Error Handling**: Tests verify proper error handling and exception cases
5. **Integration Ready**: Tests are compatible with CI/CD pipelines

## Adding New Tests

When adding new functionality:

1. Create test file in appropriate directory
2. Follow naming convention: `test_<module_name>.py`
3. Use descriptive test names: `test_<functionality>_<scenario>`
4. Include docstrings explaining what each test validates
5. Mock external dependencies (MLflow, file I/O, network requests)

## Dependencies

Test dependencies are included in `pyproject.toml`:
- pytest >= 8.4.2
- Additional test utilities as needed

## Notes

- Tests use mocking extensively to avoid external dependencies
- Temporary directories are used for file I/O tests
- Environment variables are reset between tests to ensure isolation
- FastAPI TestClient is used for API endpoint testing
