# Refactored: models/components/builder.py
# -*- coding: utf-8 -*-
"""
Model Building Utilities - Provides common functions for constructing model blocks.

Refactoring Goals Achieved:
- Renamed functions for clarity (create_ -> build_).
- Integrated `get_normalization` and `get_activation` from unified components.
- Simplified residual block logic.
- Removed `create_attention_block` (use AttentionBlock class directly).
- Added Docstrings and Type Hinting.
"""

import tensorflow as tf
from typing import List, Optional, Callable

# Import unified components
from models.components.unified import (
    create_layer_pipeline,
    get_normalization,
    get_activation
)

# --- Building Blocks ---

def build_residual_block(input_tensor: tf.Tensor,
                        filters: int,
                        kernel_size: int = 3,
                        activation: str = 'relu',
                        dropout_rate: float = 0.0,
                        use_batch_norm: bool = True,
                        l2_reg: float = 1e-4,
                        name_prefix: str = "res_block") -> tf.Tensor:
    """Builds a standard residual block (Conv -> Norm -> Act -> Conv -> Norm -> Act + Shortcut).

    Args:
        input_tensor: The input tensor.
        filters: Number of filters for the convolutional layers.
        kernel_size: Kernel size for convolutions.
        activation: Activation function name (e.g., 'relu', 'swish', 'gelu').
        dropout_rate: Dropout rate after activations.
        use_batch_norm: Whether to use Batch Normalization (True) or Layer Normalization (False).
        l2_reg: L2 regularization factor for kernels.
        name_prefix: Prefix for layer names.

    Returns:
        The output tensor of the residual block.
    """
    # Shortcut connection
    shortcut = input_tensor
    input_filters = input_tensor.shape[-1]

    # Projection shortcut if dimensions don't match
    if input_filters != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters, kernel_size=1, padding='same', use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name=f"{name_prefix}_shortcut_conv"
        )(shortcut)
        # Optional: Add normalization to the shortcut path as well
        # shortcut = get_normalization('batch' if use_batch_norm else 'layer', name=f"{name_prefix}_shortcut_norm")(shortcut)


    # --- Residual Path ---
    # Layer 1
    x = tf.keras.layers.Conv1D(
        filters, kernel_size=kernel_size, padding='same', use_bias=not use_batch_norm, # Bias often redundant with BN
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name=f"{name_prefix}_conv1"
    )(input_tensor)
    if use_batch_norm: x = get_normalization('batch', name=f"{name_prefix}_norm1")(x)
    x = get_activation(activation)(x)
    if dropout_rate > 0: x = tf.keras.layers.SpatialDropout1D(dropout_rate, name=f"{name_prefix}_drop1")(x)

    # Layer 2
    x = tf.keras.layers.Conv1D(
        filters, kernel_size=kernel_size, padding='same', use_bias=not use_batch_norm,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name=f"{name_prefix}_conv2"
    )(x)
    if use_batch_norm: x = get_normalization('batch', name=f"{name_prefix}_norm2")(x)
    # Note: Activation typically applied *after* adding the shortcut in modern resnets

    # Add shortcut
    x = tf.keras.layers.Add(name=f"{name_prefix}_add")([shortcut, x])

    # Final Activation (common practice)
    x = get_activation(activation)(x)
    # Optional: Final LayerNorm after activation
    # x = get_normalization('layer', name=f"{name_prefix}_final_norm")(x)

    return x


def shake_shake_regularization(x1: tf.Tensor, x2: tf.Tensor, training: tf.Tensor) -> tf.Tensor:
    """Applies Shake-Shake regularization.

    Randomly scales inputs during training, averages during inference.

    Args:
        x1: Output tensor from the first branch.
        x2: Output tensor from the second branch.
        training: Boolean tensor indicating training phase.

    Returns:
        The regularized output tensor.
    """
    if not isinstance(training, tf.Tensor):
        training = tf.constant(training, dtype=tf.bool) # Ensure it's a tensor

    input_dtype = x1.dtype

    @tf.function # Apply tf.function for potential optimization
    def shake_shake_train():
        batch_size = tf.shape(x1)[0]
        # Forward pass random alpha
        alpha_fwd = tf.random.uniform([batch_size, 1, 1], minval=0.0, maxval=1.0, dtype=input_dtype)
        # Backward pass random alpha
        alpha_bwd = tf.random.uniform([batch_size, 1, 1], minval=0.0, maxval=1.0, dtype=input_dtype)
        # Combine with stop_gradient to ensure correct backpropagation
        return tf.stop_gradient(alpha_bwd - alpha_fwd) * x1 + alpha_fwd * x1 + \
               tf.stop_gradient((1.0 - alpha_bwd) - (1.0 - alpha_fwd)) * x2 + (1.0 - alpha_fwd) * x2

    @tf.function
    def shake_shake_inference():
        # Average during inference
        return 0.5 * x1 + 0.5 * x2

    # Use tf.cond to switch between training and inference
    output = tf.cond(
        training,
        shake_shake_train,
        shake_shake_inference
    )
    return output