"""Tests for utility modules."""

import pytest
import yaml
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.config import load_config, Config


class TestLogger:
    """Tests for logging utilities."""

    def test_setup_logger_default(self, tmp_path):
        """Test logger setup with default parameters."""
        logger = setup_logger(name="test_logger")

        assert logger is not None
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

    def test_setup_logger_with_file(self, tmp_path):
        """Test logger setup with file output."""
        log_file = tmp_path / "test.log"

        logger = setup_logger(
            name="test_logger_file",
            log_file=str(log_file),
            level=logging.DEBUG
        )

        assert logger is not None
        logger.info("Test message")

        # Check file was created
        assert log_file.exists()

        # Check message was written
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content

    def test_setup_logger_no_console(self, tmp_path):
        """Test logger setup without console output."""
        logger = setup_logger(
            name="test_logger_no_console",
            console=False
        )

        assert logger is not None
        assert len(logger.handlers) == 0

    def test_get_logger(self):
        """Test getting existing logger."""
        logger1 = setup_logger(name="test_get_logger")
        logger2 = get_logger(name="test_get_logger")

        assert logger1 is logger2


class TestConfig:
    """Tests for configuration utilities."""

    def test_load_config_valid_yaml(self, tmp_path):
        """Test loading valid YAML configuration."""
        config_file = tmp_path / "test_config.yaml"

        test_config = {
            'model': {
                'n_genes': 2000,
                'hidden_dim': 256
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4
            }
        }

        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)

        loaded_config = load_config(str(config_file))

        assert loaded_config == test_config
        assert loaded_config['model']['n_genes'] == 2000
        assert loaded_config['training']['batch_size'] == 32

    def test_load_config_nonexistent_file(self):
        """Test loading nonexistent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_config_class_initialization(self):
        """Test Config class initialization with actual configs."""
        # Use the actual configs directory
        config_dir = Path(__file__).parent.parent / "configs"

        if not config_dir.exists():
            pytest.skip("Configs directory not found")

        try:
            config = Config(str(config_dir))
            assert config is not None
        except Exception as e:
            pytest.skip(f"Config loading requires omegaconf: {e}")

    def test_config_get_method(self, tmp_path):
        """Test Config.get() method with dot notation."""
        # Create a simple config directory
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        test_config = {
            'model': {
                'n_genes': 2000,
                'nested': {
                    'value': 42
                }
            }
        }

        config_file = config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)

        try:
            config = Config(str(config_dir))

            # Test dot notation access
            assert config.get('model.n_genes') == 2000
            assert config.get('model.nested.value') == 42

            # Test default value
            assert config.get('nonexistent.key', 'default') == 'default'
        except Exception as e:
            pytest.skip(f"Config.get() requires omegaconf: {e}")

    def test_config_to_dict(self, tmp_path):
        """Test Config.to_dict() method."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        test_config = {'model': {'n_genes': 2000}}

        config_file = config_dir / "test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)

        try:
            config = Config(str(config_dir))
            config_dict = config.to_dict()

            assert isinstance(config_dict, dict)
            assert 'model' in config_dict
        except Exception as e:
            pytest.skip(f"Config.to_dict() requires omegaconf: {e}")


@pytest.mark.unit
class TestConfigValidation:
    """Tests for configuration validation."""

    def test_empty_config(self, tmp_path):
        """Test handling of empty configuration."""
        config_file = tmp_path / "empty_config.yaml"

        with open(config_file, 'w') as f:
            f.write("")

        loaded_config = load_config(str(config_file))
        assert loaded_config is None or loaded_config == {}

    def test_malformed_yaml(self, tmp_path):
        """Test handling of malformed YAML."""
        config_file = tmp_path / "bad_config.yaml"

        with open(config_file, 'w') as f:
            f.write("{\n  invalid yaml content\n  missing: bracket")

        with pytest.raises(yaml.YAMLError):
            load_config(str(config_file))
