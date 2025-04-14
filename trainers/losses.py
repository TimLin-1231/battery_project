# New File: trainers/losses.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Loss Functions for Battery Aging Prediction Models.
"""
import tensorflow as tf
from typing import Tuple, Optional

# --- Logging Setup ---
try:
    from core.logging import LoggerFactory
    logger = LoggerFactory.get_logger("trainers.losses")
except ImportError: # pragma: no cover
    import logging
    logger = logging.getLogger("trainers.losses")
    if not logger.handlers: logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)


class EnhancedPhysicsLoss(tf.keras.losses.Loss):
    """Enhanced Physics-Informed Loss for PINN models.

    Combines standard prediction loss (e.g., MSE) with physics-based
    regularization terms derived from the model's predicted physical parameters.
    Supports weighting different loss components and output channels.
    """

    def __init__(self,
                 physics_weight: float = 0.1,
                 data_loss_weight: float = 1.0,
                 consistency_weight: float = 0.01, # Weight for parameter consistency across cells
                 smoothness_weight: float = 0.01, # Weight for prediction smoothness
                 output_channel_weights: Optional[list[float]] = None,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='enhanced_physics_loss'):
        """Initializes the enhanced physics loss.

        Args:
            physics_weight: Weight for physics-based regularization terms (e.g., tau order).
            data_loss_weight: Weight for the primary data fitting loss (e.g., MSE).
            consistency_weight: Weight for consistency regularization on non-shared RC params.
            smoothness_weight: Weight for temporal smoothness regularization on predictions.
            output_channel_weights: List of weights for each output channel in the data loss.
                                     Defaults to [1.0, 0.5, ...] if None.
            reduction: Type of tf.keras.losses.Reduction to apply.
            name: Optional name for the loss instance.
        """
        super().__init__(reduction=reduction, name=name)
        self.physics_weight = tf.constant(physics_weight, dtype=tf.float32)
        self.data_loss_weight = tf.constant(data_loss_weight, dtype=tf.float32)
        self.consistency_weight = tf.constant(consistency_weight, dtype=tf.float32)
        self.smoothness_weight = tf.constant(smoothness_weight, dtype=tf.float32)
        self.output_channel_weights = output_channel_weights

        logger.info(f"Initialized EnhancedPhysicsLoss with weights: "
                    f"Data={data_loss_weight:.3f}, Physics={physics_weight:.3f}, "
                    f"Consistency={consistency_weight:.3f}, Smoothness={smoothness_weight:.3f}")

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred_and_params: Tuple[tf.Tensor, Dict[str, tf.Tensor]]) -> tf.Tensor:
        """Calculates the total loss.

        Args:
            y_true: Ground truth tensor [batch, seq_len, n_outputs].
            y_pred_and_params: A tuple containing:
                y_pred: Model predictions [batch, seq_len, n_outputs].
                rc_params: Dictionary of predicted RC parameters from the physics layer.
                           Parameter shapes can be [batch, seq_len] or scalar depending
                           on the RC layer implementation.

        Returns:
            Scalar total loss tensor.
        """
        y_pred, rc_params = y_pred_and_params

        # Ensure compatible dtypes (compute loss in float32 for stability)
        y_true_f32 = tf.cast(y_true, tf.float32)
        y_pred_f32 = tf.cast(y_pred, tf.float32)

        # 1. Data Loss (Weighted MSE)
        if self.output_channel_weights:
            n_outputs = tf.shape(y_true_f32)[-1]
            # Ensure weights match output dimension
            weights = tf.constant(self.output_channel_weights[:n_outputs], dtype=tf.float32)
            weights = tf.reshape(weights, [1, 1, n_outputs]) # Reshape for broadcasting
            squared_errors = tf.square(y_true_f32 - y_pred_f32) * weights
        else:
            squared_errors = tf.square(y_true_f32 - y_pred_f32)
        mse_loss = tf.reduce_mean(squared_errors) # Mean over batch, seq_len, outputs

        # 2. Physics Constraint Loss (e.g., Tau order: tau1 < tau2 < tau3)
        tau_loss = 0.0
        taus = sorted([k for k in rc_params if k.startswith('tau')], key=lambda k: int(k[3:]))
        for i in range(len(taus) - 1):
             tau_i = tf.cast(rc_params[taus[i]], tf.float32)
             tau_j = tf.cast(rc_params[taus[i+1]], tf.float32)
             # Penalize if tau_i >= tau_j
             tau_loss += tf.reduce_mean(tf.maximum(0.0, tau_i - tau_j))
        # Normalize by number of comparisons
        if len(taus) > 1: tau_loss /= (len(taus) - 1)

        # 3. Parameter Consistency Loss (for non-shared parameters)
        # This requires RC params to have shape [batch, seq_len] or similar
        # If params are scalar (shared), this loss will be zero.
        consistency_loss = 0.0
        if self.consistency_weight > 0:
            param_variances = []
            for name, param in rc_params.items():
                 # Check if parameter varies across batch/time (heuristic for non-shared)
                 if isinstance(param, tf.Tensor) and param.shape.rank > 0:
                      # Calculate variance across the batch dimension (assuming [batch, seq_len])
                      # Or variance across cells if shape is [num_cells] -> needs info
                      # Simple approach: variance over all non-singleton dimensions
                      non_singleton_axes = [i for i, dim in enumerate(param.shape) if dim > 1]
                      if non_singleton_axes:
                          variance = tf.math.reduce_variance(tf.cast(param, tf.float32), axis=non_singleton_axes)
                          param_variances.append(tf.reduce_mean(variance)) # Mean variance if multiple axes varied

            if param_variances:
                consistency_loss = tf.add_n(param_variances) / len(param_variances) # Average variance

        # 4. Prediction Smoothness Loss (Temporal Regularization)
        smoothness_loss = 0.0
        if self.smoothness_weight > 0 and tf.shape(y_pred_f32)[1] > 1: # Need sequence length > 1
            # First-order difference
            diff1 = y_pred_f32[:, 1:, :] - y_pred_f32[:, :-1, :]
            smoothness_loss += tf.reduce_mean(tf.square(diff1))
            # Optional: Second-order difference
            if tf.shape(y_pred_f32)[1] > 2:
                 diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
                 smoothness_loss += 0.5 * tf.reduce_mean(tf.square(diff2))

        # Combine losses with weights
        total_loss = (self.data_loss_weight * mse_loss +
                      self.physics_weight * tau_loss +
                      self.consistency_weight * consistency_loss +
                      self.smoothness_weight * smoothness_loss)

        # Log individual components for monitoring (if needed via callbacks or metrics)
        # tf.summary.scalar('mse_loss', mse_loss, step=...) # Need step info

        return total_loss

    def get_config(self):
        """Returns the loss configuration."""
        base_config = super().get_config()
        return {
            **base_config,
            "physics_weight": float(self.physics_weight.numpy()),
            "data_loss_weight": float(self.data_loss_weight.numpy()),
            "consistency_weight": float(self.consistency_weight.numpy()),
            "smoothness_weight": float(self.smoothness_weight.numpy()),
            "output_channel_weights": self.output_channel_weights,
        }