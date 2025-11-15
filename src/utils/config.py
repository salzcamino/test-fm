"""Configuration utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_all_configs(config_dir: str = "configs") -> Dict[str, Any]:
    """Load all configuration files and merge them.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        Merged configuration dictionary
    """
    config_dir = Path(config_dir)

    # Load individual configs
    model_config = load_config(config_dir / "model_config.yaml")
    training_config = load_config(config_dir / "training_config.yaml")
    data_config = load_config(config_dir / "data_config.yaml")

    # Merge configurations
    full_config = {
        **model_config,
        **training_config,
        **data_config
    }

    return full_config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


class Config:
    """Configuration class using OmegaConf."""

    def __init__(self, config_dir: str = "configs"):
        """Initialize configuration.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config = self._load_configs()

    def _load_configs(self) -> OmegaConf:
        """Load all configuration files."""
        configs = []

        # Load all YAML files in config directory
        for config_file in sorted(self.config_dir.glob("*.yaml")):
            cfg = OmegaConf.load(config_file)
            configs.append(cfg)

        # Merge all configs
        merged_config = OmegaConf.merge(*configs)
        return merged_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self.config, resolve=True)

    def save(self, save_path: str):
        """Save configuration to file."""
        OmegaConf.save(self.config, save_path)
