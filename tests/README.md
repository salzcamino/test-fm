# Test Suite

Comprehensive test suite for the scRNA-seq Foundation Model package.

## Overview

This directory contains automated tests covering all major components of the package:

- **test_utils.py** - Tests for utility modules (logging, configuration)
- **test_losses.py** - Tests for loss functions (MLM, contrastive, combined)
- **test_metrics.py** - Tests for evaluation metrics (clustering, classification, reconstruction)
- **test_models.py** - Tests for model architectures (encoder, transformer, foundation model)
- **test_data.py** - Tests for data loading and preprocessing

## Running Tests

### Install Test Dependencies

```bash
# Install all dependencies including test tools
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Or install package with test dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run in parallel (faster)
pytest tests/ -n auto
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only smoke tests (quick checks)
pytest tests/ -m smoke

# Run tests requiring PyTorch
pytest tests/ -m requires_torch

# Run tests requiring data dependencies
pytest tests/ -m requires_data
```

### Run Specific Test Files

```bash
# Test utilities only
pytest tests/test_utils.py -v

# Test models only
pytest tests/test_models.py -v

# Test specific test class
pytest tests/test_losses.py::TestMaskedLMLoss -v

# Test specific test function
pytest tests/test_losses.py::TestMaskedLMLoss::test_mlm_loss_initialization -v
```

## Test Markers

Tests are marked with the following categories:

- `@pytest.mark.unit` - Fast unit tests for individual components
- `@pytest.mark.integration` - Integration tests for multiple components
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.requires_torch` - Tests requiring PyTorch
- `@pytest.mark.requires_data` - Tests requiring data dependencies (anndata, scanpy)
- `@pytest.mark.smoke` - Quick smoke tests to verify basic functionality

## Test Structure

Each test file follows this structure:

```python
import pytest
from src.module import ClassToTest

class TestClassName:
    """Tests for ClassName."""

    def test_functionality(self):
        """Test specific functionality."""
        # Arrange
        obj = ClassToTest()

        # Act
        result = obj.method()

        # Assert
        assert result is not None
```

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_expression_data` - Sample expression matrix (100x50)
- `small_expression_data` - Small expression matrix (10x20)
- `mock_adata` - Mock AnnData object
- `small_mock_adata` - Small mock AnnData object
- `sample_config` - Sample configuration dictionary
- `torch_available` - Check if PyTorch is available
- `skip_if_no_torch` - Skip test if PyTorch not installed

## Coverage Goals

Target coverage: **> 80%**

Current coverage can be checked with:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Writing New Tests

When adding new tests:

1. **Create test file**: `tests/test_<module>.py`
2. **Import module**: `from src.<module> import <Class>`
3. **Create test class**: `class Test<ClassName>:`
4. **Write test methods**: `def test_<functionality>():`
5. **Use fixtures**: Add fixtures from `conftest.py` as parameters
6. **Mark appropriately**: Add `@pytest.mark.<category>` decorator
7. **Document**: Add docstring explaining what is tested

Example:

```python
import pytest
from src.models.model import scRNAFoundationModel

@pytest.mark.requires_torch
class TestscRNAFoundationModel:
    """Tests for scRNA Foundation Model."""

    @pytest.fixture(autouse=True)
    def setup(self, skip_if_no_torch):
        """Setup for tests requiring torch."""
        import torch
        self.torch = torch
        self.scRNAFoundationModel = scRNAFoundationModel

    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = self.scRNAFoundationModel(
            n_genes=100,
            hidden_dim=128
        )
        assert model is not None
```

## Continuous Integration

Tests run automatically on GitHub Actions:

- **On push** to main/master/develop branches
- **On pull requests** to main/master/develop branches
- **Multiple Python versions**: 3.8, 3.9, 3.10, 3.11
- **Coverage upload** to Codecov

See `.github/workflows/tests.yml` for CI configuration.

## Troubleshooting

### Tests fail with "ModuleNotFoundError"

```bash
# Install package in development mode
pip install -e .
```

### Tests fail with "pytest: command not found"

```bash
# Install pytest
pip install pytest
```

### Tests skipped with "PyTorch not installed"

```bash
# Install PyTorch
pip install torch
```

### Tests skipped with "anndata not installed"

```bash
# Install data dependencies
pip install anndata scanpy
```

## Test Statistics

- **Total test files**: 5
- **Total test classes**: ~15
- **Total test functions**: ~80+
- **Coverage target**: >80%
- **Test categories**: 6 markers

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
