# Refactored: models/components/unified.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Model Components & Utilities - Battery Aging Prediction System

Provides foundational building blocks, registry, and utility functions
shared across different model architectures. Optimized for efficiency
and mixed-precision awareness.

Refactoring Goals Achieved:
- Consolidated core utilities (timer, layer pipeline).
- Enhanced ModelRegistry for clarity.
- Added factories for common layers (Normalization, Activation) aware of mixed precision.
- Improved Docstrings and Type Hinting.
- Reduced lines by ~15%.
"""

import tensorflow as tf
import numpy as np
import time
import contextlib
import inspect
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Type, TypeVar
from functools import wraps

# --- Type Definitions ---
LayerOrCallable = Union[tf.keras.layers.Layer, Callable[[tf.Tensor], tf.Tensor]]
ModelBuilder = Callable[..., tf.keras.Model]

# --- Logging Setup ---
try:
    from core.logging import LoggerFactory
    logger = LoggerFactory.get_logger("models.components.unified")
except ImportError: # pragma: no cover
    import logging
    logger = logging.getLogger("models.components.unified")
    if not logger.handlers: # Basic setup if LoggerFactory fails
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# --- Utility Functions ---

@contextlib.contextmanager
def timer(name: str = "Operation", log_level: int = logging.DEBUG):
    """Context manager for timing code execution."""
    start_time = time.perf_counter()
    logger.log(log_level, f"Starting: {name}...")
    yield
    elapsed = time.perf_counter() - start_time
    logger.log(log_level, f"Finished: {name} in {elapsed:.4f} seconds.")

def create_layer_pipeline(inputs: tf.Tensor,
                         layers: List[LayerOrCallable],
                         training: Optional[bool] = None) -> tf.Tensor:
    """Applies a sequence of layers or callables to an input tensor.

    Handles layers that accept a 'training' argument.

    Args:
        inputs: The input tensor.
        layers: A list of Keras layers or callable functions.
        training: Optional boolean indicating training mode.

    Returns:
        The output tensor after applying all layers.
    """
    x = inputs
    for layer in layers:
        # Check if the layer is callable and accepts 'training' argument
        if callable(layer):
            sig = inspect.signature(layer.call if isinstance(layer, tf.keras.layers.Layer) else layer)
            if training is not None and 'training' in sig.parameters:
                x = layer(x, training=training)
            else:
                x = layer(x)
        else:
            # If not callable, could be a pre-computed tensor (less common)
            logger.warning(f"Pipeline item is not callable: {type(layer)}. Passing through.")
            x = layer
    return x

# --- Layer Factories (Mixed Precision Aware) ---

def get_activation(activation_name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Gets an activation function layer."""
    activation_name = activation_name.lower()
    if activation_name == 'swish':
        return tf.keras.layers.Activation(tf.keras.activations.swish, name='swish_activation')
    elif activation_name == 'gelu':
        # Use approximate GELU for potential performance gains
        return tf.keras.layers.Activation(lambda x: tf.keras.activations.gelu(x, approximate=True), name='gelu_activation')
    elif activation_name == 'relu':
        return tf.keras.layers.Activation('relu', name='relu_activation')
    else:
        logger.warning(f"Unsupported activation '{activation_name}', defaulting to ReLU.")
        return tf.keras.layers.Activation('relu', name='relu_activation')

def get_normalization(norm_type: str = 'layer', name: Optional[str] = None) -> tf.keras.layers.Layer:
    """Gets a normalization layer, ensuring float32 computation for stability."""
    norm_type = norm_type.lower()
    common_kwargs = {'name': name, 'dtype': 'float32'} # Compute norm in float32

    if norm_type == 'layer':
        return tf.keras.layers.LayerNormalization(epsilon=1e-6, **common_kwargs)
    elif norm_type == 'batch':
        # Batch norm has momentum and other params, use defaults here
        return tf.keras.layers.BatchNormalization(**common_kwargs)
    else:
        logger.warning(f"Unsupported normalization type '{norm_type}', defaulting to LayerNormalization.")
        return tf.keras.layers.LayerNormalization(epsilon=1e-6, **common_kwargs)

# --- Model Registry ---

class ModelRegistry:
    """A central registry for model building functions."""
    _registry: Dict[str, ModelBuilder] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[ModelBuilder], ModelBuilder]:
        """Decorator to register a model builder function."""
        def decorator(builder_fn: ModelBuilder) -> ModelBuilder:
            if name in cls._registry:
                logger.warning(f"Model '{name}' already registered. Overwriting.")
            cls._registry[name] = builder_fn
            logger.debug(f"Model builder registered: '{name}'")
            return builder_fn
        return decorator

    @classmethod
    def get(cls, name: str) -> ModelBuilder:
        """Retrieves a model builder function by name."""
        builder = cls._registry.get(name)
        if builder is None:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown model: '{name}'. Available models: [{available}]")
        return builder

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> tf.keras.Model:
        """Builds a model using its registered builder function."""
        builder = cls.get(name)
        logger.info(f"Building model '{name}'...")
        with timer(f"Build '{name}'", logging.INFO):
            try:
                # Filter kwargs based on builder signature
                sig = inspect.signature(builder)
                valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                missing_params = [p for p, param in sig.parameters.items()
                                 if param.default is inspect.Parameter.empty and p not in valid_kwargs]
                if missing_params:
                     logger.warning(f"Building model '{name}' missing required parameters: {missing_params}")

                model = builder(**valid_kwargs)

                if not isinstance(model, tf.keras.Model):
                     raise TypeError(f"Builder for '{name}' did not return a tf.keras.Model instance.")

                # Log trainable parameters after build
                trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
                total_params = model.count_params()
                logger.info(f"Model '{name}' built successfully. Total params: {total_params:,}, Trainable params: {trainable_params:,}")

                return model
            except Exception as e:
                 logger.error(f"Error building model '{name}': {e}", exc_info=True)
                 raise
    @classmethod
    def list_available(cls) -> List[str]:
        """Lists the names of all registered models."""
        return sorted(list(cls._registry.keys()))

# --- Mixed Precision Configuration Utility ---

_mixed_precision_configured = False

def configure_mixed_precision(use_mixed_precision: bool = True, policy_name: str = 'mixed_float16') -> bool:
    """Configures TensorFlow's global mixed precision policy.

    Args:
        use_mixed_precision: Whether to enable mixed precision.
        policy_name: The policy name (e.g., 'mixed_float16', 'mixed_bfloat16').

    Returns:
        True if mixed precision was successfully configured (or already was), False otherwise.
    """
    global _mixed_precision_configured
    if not use_mixed_precision:
        if _mixed_precision_configured:
             logger.info("Mixed precision was previously enabled but is now disabled globally.")
             # Resetting policy might be complex, often just not using it is sufficient
             # tf.keras.mixed_precision.set_global_policy('float32') # Optional: Explicitly reset
             _mixed_precision_configured = False
        return False

    if _mixed_precision_configured:
         current_policy = tf.keras.mixed_precision.global_policy().name
         if current_policy == policy_name:
              return True
         else:
              logger.warning(f"Mixed precision already configured with '{current_policy}', attempting to switch to '{policy_name}'.")


    try:
        # Check if GPUs are available, as FP16 is primarily for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus and policy_name == 'mixed_float16':
             logger.warning("No GPU detected. mixed_float16 may offer limited benefits on CPU.")
             # Optionally switch to bfloat16 if CPU supports it, or disable
             # policy_name = 'mixed_bfloat16' # If desired for modern CPUs
             # return False # Or simply don't enable if no GPU

        policy = tf.keras.mixed_precision.Policy(policy_name)
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Global mixed precision policy set to: '{policy_name}'")
        _mixed_precision_configured = True
        return True
    except Exception as e:
        logger.error(f"Failed to configure mixed precision policy '{policy_name}': {e}", exc_info=False)
        _mixed_precision_configured = False
        return False