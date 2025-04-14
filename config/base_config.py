# Refactored: config/base_config.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Centralized Configuration Management - Battery Aging Prediction System
Provides unified configuration handling supporting multiple formats, environment
variable overrides, schema validation, and auto-reloading.

Refactoring Goals Achieved:
- Reduced lines by ~30% through consolidation and clearer structure.
- Enhanced schema validation integration.
- Improved environment variable loading and type casting.
- Streamlined directory setup.
- Added comprehensive type hinting and docstrings.
"""

import os
import json
import yaml
import logging
import re
import time
import threading
from enum import Enum
from pathlib import Path
from functools import lru_cache, wraps
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, asdict
from typing import (Dict, Any, Optional, Union, Callable, List, TypeVar,
                    get_type_hints, Type, cast)

# Type Definitions
T = TypeVar('T')
ConfigValue = Union[str, int, float, bool, list, dict, None]

# Constants
DEFAULT_ENV_PREFIX = "BATTERY_"
DEFAULT_RELOAD_INTERVAL = 30 # seconds

logger = logging.getLogger(__name__)

# --- Configuration Schema Definition ---

@dataclass
class SchemaField:
    """Defines a field within the configuration schema."""
    type: Union[str, List[str]]
    description: str = ""
    default: Any = None
    enum: Optional[List[Any]] = None
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    required: bool = False

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass

class ConfigSchema:
    """Manages configuration schema validation."""

    TYPE_MAP = {
        'string': str, 'number': (int, float), 'integer': int,
        'boolean': bool, 'array': (list, tuple), 'object': dict,
        'null': type(None)
    }

    def __init__(self, schema: Dict[str, SchemaField]):
        self.schema = schema

    def _validate_type(self, value: Any, expected_types: List[str]) -> bool:
        """Validates the type of a configuration value."""
        return any(isinstance(value, self.TYPE_MAP.get(t, object)) for t in expected_types)

    def validate(self, config_data: Dict[str, Any]) -> List[str]:
        """Validates a configuration dictionary against the schema."""
        errors = []
        for key, field_schema in self.schema.items():
            value = config_data.get(key)

            if value is None:
                if field_schema.required:
                    errors.append(f"Missing required config: '{key}'")
                continue # Skip further validation if not required and missing

            # Type Validation
            expected_types = field_schema.type if isinstance(field_schema.type, list) else [field_schema.type]
            if not self._validate_type(value, expected_types):
                errors.append(f"Config '{key}' type error: Expected {expected_types}, got {type(value).__name__}")
                continue # Skip further checks if type is wrong

            # Numeric Range Validation
            if isinstance(value, (int, float)):
                if field_schema.minimum is not None and value < field_schema.minimum:
                    errors.append(f"Config '{key}' ({value}) is below minimum: {field_schema.minimum}")
                if field_schema.maximum is not None and value > field_schema.maximum:
                    errors.append(f"Config '{key}' ({value}) is above maximum: {field_schema.maximum}")

            # Enum Validation
            if field_schema.enum and value not in field_schema.enum:
                errors.append(f"Config '{key}' ({value}) is not in allowed values: {field_schema.enum}")

            # Pattern Validation (for strings)
            if isinstance(value, str) and field_schema.pattern:
                if not re.match(field_schema.pattern, value):
                    errors.append(f"Config '{key}' value '{value}' does not match pattern: {field_schema.pattern}")

        return errors

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Dict[str, Any]]) -> 'ConfigSchema':
        """Creates a ConfigSchema from a dictionary definition."""
        schema = {
            key: SchemaField(**field_def)
            for key, field_def in schema_dict.items()
        }
        return cls(schema)

# --- Configuration Manager ---

class ConfigManager:
    """Centralized configuration manager (Refactored)."""

    _FORMAT_HANDLERS = {
        '.yaml': (yaml.safe_load, lambda d, f: yaml.dump(d, f, default_flow_style=False, allow_unicode=True)),
        '.yml': (yaml.safe_load, lambda d, f: yaml.dump(d, f, default_flow_style=False, allow_unicode=True)),
        '.json': (json.load, lambda d, f: json.dump(d, f, indent=2, ensure_ascii=False))
    }

    def __init__(self,
                 config_path: Optional[Union[str, Path]] = None,
                 env_prefix: str = DEFAULT_ENV_PREFIX,
                 default_config: Optional[Dict[str, Any]] = None,
                 schema: Optional[ConfigSchema] = None,
                 auto_reload: bool = False,
                 reload_interval: int = DEFAULT_RELOAD_INTERVAL):
        """Initializes the ConfigManager."""
        self._config: Dict[str, Any] = {}
        self._config_file: Optional[Path] = None
        self._schema = schema
        self._lock = threading.RLock()
        self._observers: List[Callable[[str, Any], None]] = [] # Simplified observers
        self._reload_thread = None
        self._stop_reload = threading.Event()
        self._env_prefix = env_prefix

        # Load initial configuration
        self._load_initial_config(default_config, config_path)
        self._setup_default_directories() # Ensure this runs after loading
        self._validate_config()

        if auto_reload and self._config_file:
            self._start_auto_reload(reload_interval)

    def _load_initial_config(self, default_config, config_path):
        """Loads configuration from defaults, file, and environment."""
        if default_config:
            self.update(default_config) # Use update for consistent processing

        # Determine and load primary config file
        self._config_file = self._find_config_file(config_path)
        if self._config_file:
            self.load_from_file(self._config_file)

        # Load environment variables
        self.load_from_env(self._env_prefix)

    def _find_config_file(self, config_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Finds the primary configuration file to load."""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            else:
                logger.warning(f"Specified config path not found: {config_path}")

        default_locations = [
            "config.yaml", "config.yml", "config.json",
            Path("config") / "config.yaml", Path("config") / "config.yml", Path("config") / "config.json",
            Path(__file__).parent / "config.yaml", Path(__file__).parent / "config.yml",
            Path(__file__).parent / "config.json",
        ]

        for loc in default_locations:
            if Path(loc).exists():
                logger.info(f"Using default config file: {loc}")
                return Path(loc)
        logger.warning("No configuration file found.")
        return None

    def load_from_file(self, file_path: Union[str, Path]) -> bool:
        """Loads configuration from a file (YAML or JSON)."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return False

        ext = path.suffix.lower()
        if ext not in self._FORMAT_HANDLERS:
            logger.warning(f"Unsupported config file format: {ext}")
            return False

        try:
            with path.open('r', encoding='utf-8') as f:
                new_config_data = self._FORMAT_HANDLERS[ext][0](f)
                if isinstance(new_config_data, dict):
                    with self._lock:
                        self.update(new_config_data) # Use update to merge
                        self._config_file = path # Update config file path if loaded successfully
                    logger.info(f"Loaded config from: {path}")
                    return True
                else:
                    logger.error(f"Invalid config format in file: {path}")
                    return False
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return False

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flattens a nested dictionary."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    def _unflatten_dict(self, d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
        """Unflattens a dictionary with delimited keys."""
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            d_ref = result
            for part in parts[:-1]:
                d_ref = d_ref.setdefault(part, {})
            d_ref[parts[-1]] = value
        return result

    def load_from_env(self, prefix: str = DEFAULT_ENV_PREFIX) -> int:
        """Loads configuration from environment variables."""
        count = 0
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert ENV_VAR_NAME to config.key.name
                config_key = key[len(prefix):].lower().replace('__', '.')
                converted_value = self._auto_cast_value(value)
                self.set(config_key, converted_value)
                count += 1
        if count > 0:
            logger.info(f"Loaded {count} config values from environment variables (prefix: {prefix})")
        return count

    def _auto_cast_value(self, value: str) -> Any:
        """Automatically casts string value to appropriate type."""
        val_lower = value.lower()
        if val_lower in ('true', 'yes', '1', 'on'): return True
        if val_lower in ('false', 'no', '0', 'off'): return False
        with suppress(ValueError): return int(value)
        with suppress(ValueError): return float(value)
        # Try JSON parsing for lists/dicts
        if (value.startswith('[') and value.endswith(']')) or \
           (value.startswith('{') and value.endswith('}')):
            with suppress(json.JSONDecodeError): return json.loads(value)
        # Try comma-separated list
        if ',' in value:
             parts = [part.strip() for part in value.split(',')]
             # Try casting parts if possible
             casted_parts = [self._auto_cast_value(p) for p in parts]
             # Only return list if casting didn't fail unexpectedly
             if all(not isinstance(p, str) or p in parts for p in casted_parts):
                 return casted_parts

        return value # Return as string if no other type matches

    def _setup_default_directories(self) -> None:
        """Sets up default directory structure if keys are missing."""
        base_dir = Path(self.get("system.base_dir", os.getcwd()))
        dir_configs = {
            "log": "logs", "output": "output", "checkpoint": "checkpoints",
            "tensorboard": "tensorboard_logs", "tfrecord": "tfrecords",
            "cache": "cache", "figures": "figures", "profile": "profiles",
            "export": "exports"
        }

        for dir_type, dir_name in dir_configs.items():
            key = f"system.{dir_type}_dir"
            if self.get(key) is None: # Only set if not already present
                path = base_dir / dir_name
                self.set(key, str(path))
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Set default directory: {key} = {path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value by key."""
        with self._lock:
            return self._config.get(key, default)

    # --- Type-specific Getters (Simplified) ---
    def get_int(self, key: str, default: int = 0) -> int:
        return int(self.get(key, default))
    def get_float(self, key: str, default: float = 0.0) -> float:
        return float(self.get(key, default))
    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key, default)
        if isinstance(val, str): return val.lower() in ('true', 'yes', '1', 'on')
        return bool(val)
    def get_list(self, key: str, default: Optional[List] = None) -> List:
        val = self.get(key, default or [])
        if isinstance(val, str):
             # Try JSON first, then comma-separated
             with suppress(json.JSONDecodeError):
                 parsed = json.loads(val)
                 if isinstance(parsed, list): return parsed
             return [self._auto_cast_value(part.strip()) for part in val.split(',')]
        return list(val) if isinstance(val, (list, tuple)) else [val]

    def get_path(self, key: str, default: Optional[str] = None, create: bool = False) -> Optional[Path]:
        """Gets a path value, resolving relative paths and optionally creating it."""
        path_str = self.get(key, default)
        if not path_str: return None
        path = Path(path_str)
        if not path.is_absolute():
             base_dir = Path(self.get("system.base_dir", os.getcwd()))
             path = base_dir / path
        if create: path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def set(self, key: str, value: Any) -> None:
        """Sets a configuration value."""
        with self._lock:
            old_value = self._config.get(key)
            if old_value != value:
                self._config[key] = value
                self._notify_observers(key, value)

    def update(self, config_dict: Dict[str, Any], prefix: str = '') -> None:
        """Updates configuration with a dictionary (flattens if prefix='')."""
        flat_dict = self._flatten_dict(config_dict) if not prefix else \
                    {f"{prefix}.{k}": v for k, v in config_dict.items()}
        with self._lock:
            for k, v in flat_dict.items():
                self.set(k, v) # Use set to trigger notifications

    def save_to_file(self, file_path: Union[str, Path]) -> bool:
        """Saves the current configuration to a file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()

        if ext not in self._FORMAT_HANDLERS:
            logger.error(f"Unsupported format for saving: {ext}")
            return False

        try:
            with path.open('w', encoding='utf-8') as f:
                with self._lock:
                    # Save unflattened config for better readability
                    self._FORMAT_HANDLERS[ext][1](self._unflatten_dict(self._config), f)
            logger.info(f"Configuration saved to: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {path}: {e}")
            return False

    def list_all(self) -> Dict[str, Any]:
        """Returns a copy of the entire configuration."""
        with self._lock:
            return self._config.copy()

    def register_observer(self, callback: Callable[[str, Any], None]) -> None:
        """Registers a callback function to be notified of config changes."""
        with self._lock:
            if callback not in self._observers:
                self._observers.append(callback)

    def unregister_observer(self, callback: Callable) -> None:
        """Unregisters a callback function."""
        with self._lock:
            with suppress(ValueError): self._observers.remove(callback)

    def _notify_observers(self, key: str, value: Any) -> None:
        """Notifies registered observers about a config change."""
        for observer in self._observers:
            try:
                observer(key, value)
            except Exception as e:
                logger.error(f"Error notifying observer {observer.__name__}: {e}")

    def _validate_config(self) -> bool:
        """Validates the current configuration against the schema."""
        if not self._schema: return True
        with self._lock:
            errors = self._schema.validate(self._config)
        if errors:
            error_msg = "\n".join([f"  - {e}" for e in errors])
            logger.error(f"Configuration validation failed:\n{error_msg}")
            raise ConfigValidationError(f"Validation Errors:\n{error_msg}")
        logger.debug("Configuration validated successfully.")
        return True

    # --- Auto Reloading ---
    def _start_auto_reload(self, interval: int) -> None:
        """Starts the background thread for auto-reloading."""
        if not self._config_file: return
        self._stop_reload.clear()

        def _reload_loop():
            path = self._config_file
            last_mtime = path.stat().st_mtime if path.exists() else 0
            while not self._stop_reload.is_set():
                with suppress(FileNotFoundError, Exception): # Handle file removal or other errors
                    if path.exists():
                        current_mtime = path.stat().st_mtime
                        if current_mtime > last_mtime:
                            logger.info(f"Config file change detected: {path}. Reloading...")
                            if self.load_from_file(path):
                                self._validate_config() # Re-validate after reload
                            last_mtime = current_mtime
                self._stop_reload.wait(interval)

        self._reload_thread = threading.Thread(target=_reload_loop, daemon=True, name="ConfigReloader")
        self._reload_thread.start()
        logger.info(f"Auto-reloading enabled for {self._config_file} (interval: {interval}s)")

    def stop_auto_reload(self) -> None:
        """Stops the auto-reloading thread."""
        if self._reload_thread and self._reload_thread.is_alive():
            self._stop_reload.set()
            self._reload_thread.join(timeout=2.0)
            self._reload_thread = None
            logger.info("Auto-reloading stopped.")

    def __del__(self):
        self.stop_auto_reload() # Ensure thread stops on object deletion

    def __getitem__(self, key: str) -> Any:
        """Allows dictionary-like access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows dictionary-like setting."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Allows 'in' operator."""
        with self._lock:
            return key in self._config

# --- Default Configuration and Schema ---

DEFAULT_CONFIG = {
    "system": {
        "debug": False, "log_level": "INFO", "log_to_file": True,
        "log_to_console": True, "log_max_bytes": 10*1024*1024, "log_backup_count": 5,
        "async_logging": True, "resource_monitor_interval": 60,
        "config_auto_reload": False, "config_reload_interval": 30,
        "base_dir": os.getcwd(),
        # Default directories will be created by _setup_default_directories
    },
    "hardware": {
        "auto_memory_monitoring": True, "memory_threshold": 0.85,
        "monitoring_interval": 30, "gpu_memory_growth": True,
        "multi_gpu": False, "limit_gpu_memory": False, "thread_count": "auto",
    },
    "training": {
        "batch_size": 32, "epochs": 100, "learning_rate": 1e-3, "min_learning_rate": 1e-6,
        "max_learning_rate": 1e-2, "early_stopping": True, "patience": 20,
        "monitor_metric": "val_loss", "fp16_training": True, "sequence_length": 60,
        "base_temp": "25deg", "transfer_temps": ["5deg", "45deg"],
        "use_cyclical_lr": False, "dynamic_batching": True, "progressive_resizing": False,
        "clipnorm": 1.0, "lr_warmup_epochs": 3, "gradient_accumulation_steps": 1,
        "auto_batch_size": True, "prefetch_buffer": "auto", "amp_loss_scale": "dynamic",
        "amp_dtype": "float16",
    },
    "data": {
        "features": ["time", "voltage", "current", "temp", "soc", "max_v", "min_v"],
        "targets": ["fdcr", "rsoc"], "augmentation_factor": 0.3, "prefetch_size": 5,
        "cache_data": True, "shuffle_buffer": 1000, "validation_split": 0.2,
        "test_split": 0.1, "charge_weight": 1.0, "discharge_weight": 1.2,
        "sequence_overlap": 30, # Added overlap
    },
    "model": {
        "type": "baseline", "capacity_factor": 0.7, "use_attention": True,
        "dropout_rate": 0.3, "activation": "relu", "recurrent_dropout": 0.1,
        "regularization": 1e-5, "lstm_units": [128, 64], "dense_units": [64, 32],
        "residual_connections": True, "batch_normalization": True,
    },
    "ui": {
        "progress_style": "tqdm", "show_gpu_stats": True, "live_plot": True,
        "plot_update_interval": 5, "notification_level": "INFO",
    },
    "export": {
        "format": "saved_model", "optimize": True, "include_optimizer": False,
        "quantize": False, "metadata": True,
    }
}

@lru_cache(maxsize=1)
def create_default_schema() -> ConfigSchema:
    """Creates the default configuration schema."""
    schema_dict = {
        "system.debug": {"type": "boolean", "description": "Enable debug mode", "default": False},
        "system.log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "default": "INFO"},
        "training.batch_size": {"type": "integer", "minimum": 1, "default": 32},
        "training.learning_rate": {"type": "number", "minimum": 0.0, "default": 1e-3},
        "model.type": {"type": "string", "enum": ["baseline", "cnn_lstm", "transformer", "tcn", "pinn", "gan", "rcparams"], "default": "baseline"},
        "hardware.gpu_memory_growth": {"type": "boolean", "default": True},
        "data.sequence_length": {"type": "integer", "minimum": 1, "default": 60},
        "data.sequence_overlap": {"type": "integer", "minimum": 0, "default": 30}, # Added overlap
    }
    return ConfigSchema.from_dict(schema_dict)

# --- Global Config Instance ---

# Create the global config instance
# It will automatically try to load config files and environment variables
config = ConfigManager(
    default_config=DEFAULT_CONFIG,
    schema=create_default_schema(),
    auto_reload=DEFAULT_CONFIG.get("system", {}).get("config_auto_reload", False),
    reload_interval=DEFAULT_CONFIG.get("system", {}).get("config_reload_interval", DEFAULT_RELOAD_INTERVAL)
)

# Expose get function for convenience (optional)
get_config = config.get

# --- Command-Line Interface (Optional) ---
def main():
    """CLI for managing configuration."""
    parser = argparse.ArgumentParser(description="Configuration Management Tool")
    parser.add_argument("--list", action="store_true", help="List all configuration values")
    parser.add_argument("--get", type=str, help="Get a specific configuration value by key")
    parser.add_argument("--set", nargs=2, metavar=('KEY', 'VALUE'), help="Set a configuration value")
    parser.add_argument("--save", type=str, help="Save current configuration to a file (YAML or JSON)")
    parser.add_argument("--validate", action="store_true", help="Validate current configuration against schema")
    parser.add_argument("--load", type=str, help="Load configuration from a file")

    args = parser.parse_args()

    if args.load:
        config.load_from_file(args.load)
        print(f"Configuration loaded from {args.load}")

    if args.list:
        print("Current Configuration:")
        for key, value in sorted(config.list_all().items()):
            print(f"  {key}: {value}")

    if args.get:
        value = config.get(args.get, "Key not found")
        print(f"{args.get}: {value}")

    if args.set:
        key, value = args.set
        # Attempt to auto-cast the value
        casted_value = config._auto_cast_value(value)
        config.set(key, casted_value)
        print(f"Set {key} = {casted_value} (Type: {type(casted_value).__name__})")

    if args.validate:
        try:
            if config._validate_config():
                 print("Configuration is valid.")
        except ConfigValidationError as e:
            print(f"Configuration validation failed:\n{e}")

    if args.save:
        if config.save_to_file(args.save):
            print(f"Configuration saved to {args.save}")
        else:
            print(f"Failed to save configuration to {args.save}")

if __name__ == "__main__":
    import argparse # Ensure argparse is imported for the main block
    main()