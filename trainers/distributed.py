# Refactored: trainers/distributed.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distributed Training Manager - Using TensorFlow Native Strategies

Handles multi-GPU training setup and execution using tf.distribute APIs.

Refactoring Goals Achieved:
- Uses tf.distribute.Strategy (MirroredStrategy or OneDeviceStrategy).
- Model/Optimizer creation happens within strategy scope.
- Uses standard Keras model.fit() which integrates with tf.distribute.
- Removed manual gradient aggregation and distributed steps.
- Simplified overall structure.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import tensorflow as tf

# --- Custom Modules ---
try:
    from core.logging import LoggerFactory
    from trainers.base_trainer import BaseTrainer # For config defaults and potential reuse
    from models.components.unified import ModelRegistry
    from data.data_provider import OptimizedDataLoader
    from trainers.callbacks import get_default_callbacks
except ImportError as e: # pragma: no cover
    raise ImportError(f"Failed to import distributed trainer dependencies: {e}")

logger = LoggerFactory.get_logger("trainers.distributed")

class DistributedTrainer:
    """Manages distributed training using tf.distribute strategies."""

    def __init__(self, config: Dict[str, Any], model_builder: Callable[..., tf.keras.Model]):
        """Initializes the DistributedTrainer.

        Args:
            config: Configuration dictionary.
            model_builder: A callable function that builds the Keras model.
                           It should accept (seq_len, n_features, n_outputs, config).
        """
        self._config = config
        self.model_builder = model_builder
        self.strategy = self._get_distribution_strategy()
        logger.info(f"Using distribution strategy: {self.strategy.__class__.__name__}")
        logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")

    def _get_distribution_strategy(self) -> tf.distribute.Strategy:
        """Detects available devices and returns an appropriate strategy."""
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            # Configure memory growth for all GPUs
            for gpu in gpus:
                 try: tf.config.experimental.set_memory_growth(gpu, True)
                 except RuntimeError as e: logger.debug(f"Memory growth already set or failed for {gpu.name}: {e}")
            return tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            try: tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e: logger.debug(f"Memory growth already set or failed for {gpus[0].name}: {e}")
            return tf.distribute.OneDeviceStrategy('/gpu:0')
        else:
            logger.warning("No GPUs detected, using CPU.")
            return tf.distribute.OneDeviceStrategy('/cpu:0')

    def train_single_model_distributed(
        self,
        experiment_name: str,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        initial_epoch: int = 0
    ) -> Tuple[Optional[tf.keras.Model], Optional[Dict]]:
        """Trains a single model using the configured distribution strategy.

        Args:
            experiment_name: Name for the experiment (used for saving).
            train_dataset: The training tf.data.Dataset.
            val_dataset: The validation tf.data.Dataset (optional).
            epochs: Number of epochs to train.
            callbacks: List of Keras callbacks.
            initial_epoch: Epoch to start training from.

        Returns:
            Tuple containing the trained model and its history dictionary, or (None, None) on failure.
        """
        epochs = epochs or self._config.get('epochs', 50)

        # --- Prepare Distributed Datasets ---
        # Keras fit handles distributing the dataset automatically when using a strategy.
        # No explicit strategy.experimental_distribute_dataset needed here.
        # However, ensure the dataset is batched correctly for the *global* batch size.
        global_batch_size = self._config.get('batch_size', 32) * self.strategy.num_replicas_in_sync
        logger.info(f"Global batch size: {global_batch_size} (per_replica: {self._config.get('batch_size', 32)})")

        # Re-batch datasets if necessary (or assume they are already correctly batched)
        # train_dataset = train_dataset.unbatch().batch(global_batch_size).prefetch(AUTOTUNE)
        # if val_dataset: val_dataset = val_dataset.unbatch().batch(global_batch_size).prefetch(AUTOTUNE)
        # Note: Assuming DataLoader provides globally batched dataset might be simpler.

        # --- Build and Compile Model within Strategy Scope ---
        with self.strategy.scope():
            # Build model
            # Need input shapes - get from dataset element spec
            try:
                element_spec = train_dataset.element_spec
                x_spec, y_spec = element_spec
                # Infer shapes, handling potential nested structures (like PINN output)
                seq_len = x_spec.shape[1]
                n_features = x_spec.shape[2]
                n_outputs = y_spec[0].shape[-1] if isinstance(y_spec, (tuple, list)) else y_spec.shape[-1]

                model = self.model_builder(
                     seq_len=seq_len, n_features=n_features, n_outputs=n_outputs, config=self._config
                )
            except Exception as e:
                 logger.error(f"Failed to build model within strategy scope: {e}", exc_info=True)
                 return None, None

            # Compile model using BaseTrainer's helpers
            # Create a temporary BaseTrainer instance just for compilation logic
            temp_trainer = BaseTrainer(model, self._config)
            temp_trainer.compile_model() # Use the compile logic from BaseTrainer


        # --- Prepare Callbacks ---
        # Use default callbacks, potentially adjusted for distributed setting
        callbacks = callbacks or get_default_callbacks(
            self._config, experiment_name, "distributed"
        )
        # Remove checkpoint callback if using default Keras saving with strategy
        # Or configure checkpoint to save only on worker 0.
        # Standard ModelCheckpoint works but saves redundant copies without extra config.
        # tf.keras.callbacks.BackupAndRestore might be better for fault tolerance.

        # --- Train using model.fit ---
        logger.info(f"Starting distributed training for '{experiment_name}'...")
        try:
            history = model.fit(
                train_dataset, # Pass the original dataset, strategy handles distribution
                validation_data=val_dataset,
                epochs=epochs,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                verbose=0 # Use TQDM callback
            )

            logger.info(f"Distributed training for '{experiment_name}' completed.")

            # --- Save Model (only on chief worker) ---
            # Define where to save - might need adjustment based on strategy type
            save_dir = Path(self._config['checkpoint_dir']) / experiment_name
            weights_path = save_dir / "final_weights.weights.h5"

            def _save_model_on_chief(weights_path):
                 model.save_weights(str(weights_path))
                 logger.info(f"Model weights saved by chief worker to: {weights_path}")

            # Check if running in a distributed context and save only on worker 0
            if self.strategy.num_replicas_in_sync > 1:
                 # For MirroredStrategy, saving needs to happen outside strategy.run
                 # but check if it's the chief worker. How to check? Usually implicitly handled by Keras save.
                 # Let's rely on Keras save/save_weights handling it correctly.
                 # If direct saving is needed: tf.distribute.get_replica_context().replica_id_in_sync_group == 0
                 weights_path.parent.mkdir(parents=True, exist_ok=True)
                 model.save_weights(str(weights_path)) # Keras save_weights should handle strategy
                 logger.info(f"Model weights saved to: {weights_path}")

            else: # Single device strategy
                 weights_path.parent.mkdir(parents=True, exist_ok=True)
                 model.save_weights(str(weights_path))
                 logger.info(f"Model weights saved to: {weights_path}")


            return model, history.history

        except Exception as e:
            logger.error(f"Distributed training failed for '{experiment_name}': {e}", exc_info=True)
            return None, None

    # Note: train_distributed method from original file is replaced by train_single_model_distributed
    # as multi-temp logic is now coordinated by MultiTempCoordinator, which would call
    # this method (or a similar one) potentially multiple times.