# New File: trainers/pinn_trainer.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PINN Trainer Module - Battery Aging Prediction System

Handles the training logic specific to Physics-Informed Neural Networks (PINNs),
including handling dual model outputs and custom physics-based losses.
"""
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

# --- Custom Modules ---
try:
    from config.base_config import config as global_config
    from core.logging import LoggerFactory
    from trainers.base_trainer import BaseTrainer # Inherit from BaseTrainer
    from trainers.losses import EnhancedPhysicsLoss # Import custom loss
except ImportError as e: # pragma: no cover
    raise ImportError(f"Failed to import PINN trainer dependencies: {e}")

logger = LoggerFactory.get_logger("trainers.pinn")

class PINNTrainer(BaseTrainer):
    """Trainer specialized for Physics-Informed Neural Networks."""

    def __init__(self, model: tf.keras.Model, config_override: Optional[Dict[str, Any]] = None):
        """Initializes the PINNTrainer."""
        super().__init__(model, config_override)
        logger.info("PINN Trainer initialized.")
        # Ensure model output is suitable for PINN loss
        if not isinstance(self.model.output, (list, tuple)) or len(self.model.output) < 2:
             logger.warning("PINN model output structure might be incompatible with EnhancedPhysicsLoss "
                          "(expected [y_pred, rc_params]). Ensure loss function handles the output correctly.")

    def _get_loss(self) -> Callable:
        """Overrides base method to provide the PINN-specific loss function."""
        # Create the enhanced physics loss instance from config
        loss_config = {
            'physics_weight': self._config.get('pinn_physics_weight', 0.1),
            'data_loss_weight': self._config.get('pinn_data_loss_weight', 1.0),
            'consistency_weight': self._config.get('pinn_consistency_weight', 0.01),
            'smoothness_weight': self._config.get('pinn_smoothness_weight', 0.01),
            'output_channel_weights': self._config.get('pinn_output_weights', [1.0, 0.5]), # Example weights
            'name': 'enhanced_physics_loss'
        }
        logger.info(f"Using EnhancedPhysicsLoss with config: {loss_config}")
        return EnhancedPhysicsLoss(**loss_config)

    # Override compile_model to ensure the custom loss is used correctly
    def compile_model(self, optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                      metrics: Optional[List[Any]] = None):
        """Compiles the PINN model with the EnhancedPhysicsLoss."""
        if not self.model: raise ValueError("Model not set.")

        self.optimizer = optimizer or self._get_optimizer()
        self.loss_fn = self._get_loss() # Get the PINN loss
        # Standard metrics usually apply to the *first* output (y_pred)
        self.metrics = metrics or self._get_metrics()

        # Compile with the custom loss. Keras handles applying it to the first output.
        # The loss function itself accesses the second output (params).
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn, # Use the custom loss object
            metrics=self.metrics # Metrics apply to the first output by default
        )
        self.model_compiled = True
        logger.info(f"PINN Model compiled. Optimizer: {self.optimizer.__class__.__name__}, Loss: {self.loss_fn.name}, Metrics: {[m.name if hasattr(m,'name') else m for m in self.metrics]}")
        self._log_model_summary()


    # Override train/test steps slightly if needed to handle multiple outputs for metrics explicitly
    # (though model.compiled_loss/metrics should handle it)
    @tf.function
    def _train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Performs a single PINN training step."""
        with tf.GradientTape() as tape:
            # Model returns [y_pred, rc_params]
            predictions = self.model(x, training=True)
            y_pred_main = predictions[0] # Main output for standard metrics/loss comparison

            # Loss function receives both outputs
            loss = self.loss_fn(y, predictions) # Loss function knows how to handle the tuple

            # Add regularization losses from the model itself
            if self.model.losses:
                 loss += tf.add_n(self.model.losses)

            # Mixed precision scaling
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                 scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                 scaled_loss = loss

        # Calculate and apply gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)

        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
             gradients = self.optimizer.get_unscaled_gradients(gradients)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update compiled metrics using the main prediction output
        self.model.compiled_metrics.update_state(y, y_pred_main)

        # Return metrics including loss
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss # Log the *total* calculated loss
        # Optionally log sub-components of the loss if the loss function stores them
        # if hasattr(self.loss_fn, 'last_mse_loss'): results['mse_loss'] = self.loss_fn.last_mse_loss
        # if hasattr(self.loss_fn, 'last_physics_loss'): results['physics_loss'] = self.loss_fn.last_physics_loss
        return results

    @tf.function
    def _test_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Performs a single PINN evaluation step."""
        # Model returns [y_pred, rc_params]
        predictions = self.model(x, training=False)
        y_pred_main = predictions[0] # Main output for standard metrics/loss comparison

        # Loss function receives both outputs
        loss = self.loss_fn(y, predictions) # Use the custom loss function

        # Add regularization losses
        if self.model.losses:
             loss += tf.add_n(self.model.losses)

        # Update compiled metrics using the main prediction output
        self.model.compiled_metrics.update_state(y, y_pred_main)

        # Return metrics including loss
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss # Log the *total* calculated loss
        return results

    # Inherits train, evaluate, save_model, load_weights etc. from BaseTrainer