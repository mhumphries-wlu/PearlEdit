"""Configuration and settings management."""
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

try:
    from platformdirs import user_config_dir
except ImportError:
    # Fallback if platformdirs not available
    import os
    def user_config_dir(appname: str) -> str:
        if os.name == 'nt':  # Windows
            return os.path.join(os.environ.get('APPDATA', ''), appname)
        else:  # Unix-like
            return os.path.join(os.environ.get('HOME', ''), '.config', appname)

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration operations fail."""
    pass


@dataclass
class AppSettings:
    """Application settings."""
    threshold: int = 127
    margin: int = 10
    auto_split: bool = False
    rotation_step: float = 90.0
    default_quality: int = 95
    batch_process: bool = False
    suppress_all_images_warning: bool = False
    seam_threshold: int = 140
    seam_confidence_min: float = 0.55
    seam_angle_max_deg: float = 12.0
    seam_min_length_ratio: float = 0.6
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AppSettings':
        """Create settings from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def get_config_path() -> Path:
    """
    Get the path to the configuration file.
    
    Returns:
        Path to settings.json
    """
    config_dir = Path(user_config_dir("PearlEdit"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "settings.json"


def load_settings() -> AppSettings:
    """
    Load settings from disk.
    
    Returns:
        AppSettings object with loaded settings
    """
    config_path = get_config_path()
    
    if not config_path.exists():
        logger.info("No settings file found, using defaults")
        return AppSettings()
    
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        settings = AppSettings.from_dict(data)
        logger.info(f"Loaded settings from {config_path}")
        return settings
    except Exception as e:
        logger.warning(f"Failed to load settings: {e}, using defaults")
        return AppSettings()


def save_settings(settings: AppSettings) -> None:
    """
    Save settings to disk.
    
    Args:
        settings: AppSettings object to save
        
    Raises:
        ConfigError: If save fails
    """
    config_path = get_config_path()
    
    try:
        with open(config_path, 'w') as f:
            json.dump(settings.to_dict(), f, indent=2)
        logger.info(f"Saved settings to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        raise ConfigError(f"Failed to save settings: {str(e)}") from e

