#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized and Modular Training Callbacks - Battery Aging Prediction System

Provides a set of focused callbacks for monitoring progress, resources,
handling charge/discharge specifics, TensorBoard logging, checkpoints,
learning rate scheduling, and early stopping with cooldown.

Refactoring Goals Achieved:
- Ensured Single Responsibility Principle for each callback.
- Improved ProgressCallback using tqdm.
- Optimized ResourceMonitorCallback focusing on essential metrics.
- Refined ChargeDischargeCallback.
- Standardized TensorBoard logging.
- Added EarlyStoppingWithCooldown.
- Added comprehensive Docstrings and Type Hinting.
- Consolidated common callback functionalities.
"""

import os
import time
import json
import datetime
import enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import warnings

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# Define constants
AUTOTUNE = tf.data.AUTOTUNE

# Define enums
class MemoryAlert(enum.Enum):
    """Memory utilization alert levels."""
    NORMAL = 0
    WARNING = 1
    HIGH = 2
    CRITICAL = 3

# --- Logging and Memory Utils ---
try:
    from core.logging import LoggerFactory
    from core.memory import memory_manager # Use global instance
    logger = LoggerFactory.get_logger("trainers.callbacks")
except ImportError: # pragma: no cover
    import logging
    logger = logging.getLogger("trainers.callbacks")
    if not logger.handlers: logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
    class MemoryManager: # Dummy
        def get_memory_snapshot(self): 
            class MemSnapshot:
                def __init__(self):
                    self.proc_rss_gb = 0.0
                    self.cpu_percent = 0.0
                    self.sys_percent = 0.0
                    self.proc_percent_rss = 0.0
                    self.alert_level = MemoryAlert.NORMAL
                    self.gpu_mem = []
                
                def log_summary(self, level): 
                    logger.log(level, f"Memory snapshot: {self.proc_rss_gb:.2f}GB")
            return MemSnapshot()
        def monitor_gpu(self): return {}
        def run_cleanup_strategies(self): pass
    memory_manager = MemoryManager()


# --- Base Callback Utility ---
class CallbackInputError(ValueError):
    """Custom exception for callback input errors."""
    pass


# --- Specific Callbacks ---

class TQDMProgressBar(tf.keras.callbacks.Callback):
    """Integrates TQDM progress bars with Keras training."""

    def __init__(self, epochs: Optional[int] = None, steps_per_epoch: Optional[int] = None,
                 metrics_format: str = "{:.4f}", verbose: int = 1):
        """
        Args:
            epochs: Total number of epochs (optional, can be detected from model).
            steps_per_epoch: Number of steps per epoch (optional, can be detected).
            metrics_format: Format string for metric values.
            verbose: Output verbosity level (0=silent, 1=epoch only, 2=epoch+step).
        """
        super().__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.metrics_format = metrics_format
        self.verbose = verbose
        self.epoch_bar = None
        self.step_bar = None
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        """Initialize the epoch progress bar."""
        if self.verbose <= 0:
            return
            
        self.epochs = self.epochs or self.params.get('epochs')
        if self.epochs:
            self.epoch_bar = tqdm(total=self.epochs, desc="Epochs", unit="epoch", position=0, leave=True)
        else:
            logger.warning("TQDMProgressBar: Total epochs not specified.")

    def on_epoch_begin(self, epoch, logs=None):
        """Initialize the step progress bar for current epoch."""
        self.current_epoch = epoch
        if self.verbose <= 0 or not self.epoch_bar:
            return
            
        # Try getting steps dynamically if not provided initially
        if self.steps_per_epoch is None and self.params.get('steps'):
            self.steps_per_epoch = self.params['steps']

        if self.steps_per_epoch and self.verbose > 1:
            self.step_bar = tqdm(
                total=self.steps_per_epoch, 
                desc=f"Epoch {epoch + 1}/{self.epochs}", 
                unit="step", 
                position=1, 
                leave=False
            )

    def on_train_batch_end(self, batch, logs=None):
        """Update step progress after each batch."""
        if self.verbose <= 1 or not self.step_bar:
            return
            
        logs = logs or {}
        self.step_bar.update(1)
        
        # Filter relevant metrics
        metrics = {k: v for k, v in logs.items() 
                   if k in ['loss', 'mae', 'rmse', 'r2_score']}
        
        self.step_bar.set_postfix(
            {k: f"{v:{self.metrics_format}}" for k, v in metrics.items()}
        )

    def on_epoch_end(self, epoch, logs=None):
        """Update epoch progress and display metrics."""
        if self.verbose <= 0:
            return
            
        if self.step_bar:
            self.step_bar.close()
            self.step_bar = None
            
        if self.epoch_bar:
            logs = logs or {}
            metrics = {k: v for k, v in logs.items() if not k.startswith('_')}
            self.epoch_bar.set_postfix(
                {k: f"{v:{self.metrics_format}}" for k, v in metrics.items()}
            )
            self.epoch_bar.update(1)

    def on_train_end(self, logs=None):
        """Clean up progress bars."""
        if self.verbose <= 0:
            return
            
        if self.step_bar:
            self.step_bar.close()
        if self.epoch_bar:
            self.epoch_bar.close()


class ResourceMonitorCallback(tf.keras.callbacks.Callback):
    """Monitors system resources (Memory, optional GPU) periodically."""

    def __init__(self, check_interval_batches: int = 50, mem_alert_threshold: float = 0.9):
        """
        Args:
            check_interval_batches: How often to check resources (in batches).
            mem_alert_threshold: Memory usage ratio to trigger a warning.
        """
        super().__init__()
        self.check_interval = check_interval_batches
        self.mem_alert_threshold = mem_alert_threshold
        self.batch_count = 0
        self.last_mem_check = 0.0  # GB

    def on_train_batch_end(self, batch, logs=None):
        """Checks resources at specified intervals."""
        self.batch_count += 1
        if self.batch_count % self.check_interval != 0:
            return
            
        snapshot = memory_manager.get_memory_snapshot()
        mem_change = snapshot.proc_rss_gb - self.last_mem_check
        self.last_mem_check = snapshot.proc_rss_gb

        # Log concise info periodically
        gpu_util = [g.get('utilization', 0) for g in snapshot.gpu_mem]
        gpu_util_avg = np.mean(gpu_util) if gpu_util else 0
        
        logger.debug(
            f"Resource Check (Batch {self.batch_count}): "
            f"CPU: {snapshot.cpu_percent:.1f}% | "
            f"SysMem: {snapshot.sys_percent:.1%} | "
            f"ProcMem: {snapshot.proc_rss_gb:.2f}GB ({mem_change:+.2f}GB) | "
            f"GPU Util: {gpu_util_avg:.1f}%"
        )

        if snapshot.alert_level >= MemoryAlert.CRITICAL:
            logger.warning(
                f"High memory usage detected: {snapshot.proc_rss_gb:.2f}GB "
                f"({snapshot.proc_percent_rss:.1%}). Alert: {snapshot.alert_level.name}"
            )

    def on_epoch_end(self, epoch, logs=None):
        """Logs resource summary at the end of epoch and performs GC."""
        snapshot = memory_manager.get_memory_snapshot()
        snapshot.log_summary(logging.INFO)
        # Run garbage collection at end of epoch
        memory_manager.run_cleanup_strategies()


class ChargeDischargeCallback(tf.keras.callbacks.Callback):
    """Analyzes performance separately for charge and discharge cycles."""

    def __init__(self, validation_data: tf.data.Dataset,
                 current_feature_idx: int = 2, threshold: float = 0.0,
                 eval_interval: int = 5, max_eval_samples: int = 10000):
        """
        Args:
            validation_data: The validation dataset.
                **IMPORTANT**: This dataset must be re-iterable (not a one-time iterator).
                Either use `.cache()` when creating the dataset or ensure it's constructed
                in a way that allows multiple passes. If using a dataset that can only be
                iterated once, this callback will fail after the first evaluation.
            current_feature_idx: Index of the current feature in x_batch.
            threshold: Current value threshold to separate charge/discharge.
            eval_interval: How often (in epochs) to perform the analysis.
            max_eval_samples: Max samples to use for analysis to avoid OOM.
        """
        super().__init__()
        # Validate the dataset is appropriate for multiple iterations
        if isinstance(validation_data, tf.data.Dataset):
            logger.debug("ChargeDischargeCallback initialized. Ensure validation_data is re-iterable (with cache() or similar).")
        else:
            raise CallbackInputError(
                "validation_data must be a tf.data.Dataset instance. "
                "For iterator objects, consider converting to a cached dataset first."
            )
            
        self.validation_data = validation_data
        self.current_feature_idx = current_feature_idx
        self.threshold = threshold
        self.eval_interval = eval_interval
        self.max_eval_samples = max_eval_samples
        self.charge_metrics_history: Dict[str, List] = {}
        self.discharge_metrics_history: Dict[str, List] = {}
        self.ratio_history: Dict[str, List] = {}

    def _get_data_subset(self, is_charging_target: bool) -> Optional[tf.data.Dataset]:
        """
        Filters the validation dataset for charge or discharge samples.
        
        Args:
            is_charging_target: If True, filter for charging samples, otherwise discharge.
            
        Returns:
            A filtered and batched dataset ready for evaluation, or None if filtering failed.
        """
        try:
            # Define filter function to separate charge/discharge samples
            @tf.function
            def filter_fn(x, y):
                mean_current = tf.reduce_mean(x[:, self.current_feature_idx])
                return mean_current > self.threshold if is_charging_target else mean_current <= self.threshold

            # Filter → Take limited samples → Batch → Prefetch
            filtered_ds = self.validation_data.filter(filter_fn)
            subset_ds = filtered_ds.take(self.max_eval_samples)
            
            # Get batch size from model or use default
            eval_batch_size = getattr(self.model, 'batch_size', None) or 32
            return subset_ds.batch(eval_batch_size).prefetch(AUTOTUNE)

        except Exception as e:
            logger.warning(f"Failed to create data subset for charge/discharge analysis: {e}")
            return None

    def on_epoch_end(self, epoch, logs=None):
        """Performs charge/discharge evaluation at specified intervals."""
        logs = logs or {}
        if (epoch + 1) % self.eval_interval != 0:
            return

        logger.info(f"Performing charge/discharge analysis for epoch {epoch + 1}...")

        # Get charge and discharge datasets
        charge_ds = self._get_data_subset(is_charging_target=True)
        discharge_ds = self._get_data_subset(is_charging_target=False)

        charge_results = {}
        discharge_results = {}

        # Evaluate charge data
        if charge_ds:
            try:
                charge_eval = self.model.evaluate(charge_ds, verbose=0, return_dict=True)
                charge_results = {f"val_charge_{k}": v for k, v in charge_eval.items()}
                logger.info(f"  Charge metrics: {charge_results}")
            except Exception as e:
                logger.warning(f"Charge evaluation failed: {e}")

        # Evaluate discharge data
        if discharge_ds:
            try:
                discharge_eval = self.model.evaluate(discharge_ds, verbose=0, return_dict=True)
                discharge_results = {f"val_discharge_{k}": v for k, v in discharge_eval.items()}
                logger.info(f"  Discharge metrics: {discharge_results}")
            except Exception as e:
                logger.warning(f"Discharge evaluation failed: {e}")

        # Update logs and history
        logs.update(charge_results)
        logs.update(discharge_results)

        # Store history
        for k, v in charge_results.items():
            self.charge_metrics_history.setdefault(k, []).append(v)
        for k, v in discharge_results.items():
            self.discharge_metrics_history.setdefault(k, []).append(v)

        # Calculate and store ratio
        charge_loss = charge_results.get('val_charge_loss')
        discharge_loss = discharge_results.get('val_discharge_loss')
        
        if charge_loss is not None and discharge_loss is not None and charge_loss > 1e-6:
            ratio = discharge_loss / charge_loss
            logs['val_discharge_charge_loss_ratio'] = ratio
            self.ratio_history.setdefault('loss_ratio', []).append(ratio)
            logger.info(f"  Discharge/Charge Loss Ratio: {ratio:.3f}")


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    """Custom TensorBoard callback to log additional metrics."""

    def __init__(self, log_dir: Union[str, Path], 
                 charge_discharge_callback: Optional[ChargeDischargeCallback] = None, 
                 **kwargs):
        """
        Initialize the enhanced TensorBoard callback.
        
        Args:
            log_dir: Directory to save the TensorBoard logs.
            charge_discharge_callback: Optional callback to get charge/discharge metrics.
            **kwargs: Additional arguments for the standard TensorBoard callback.
        """
        super().__init__(log_dir=str(log_dir), **kwargs)
        self.charge_discharge_callback = charge_discharge_callback

    def on_epoch_end(self, epoch, logs=None):
        """Logs standard and custom metrics."""
        logs = logs or {}
        super().on_epoch_end(epoch, logs)  # Log standard metrics

        # Log charge/discharge metrics if available
        if self.charge_discharge_callback:
            with self.writer.as_default():
                cd_metric_prefixes = ('val_charge_', 'val_discharge_', 'val_discharge_charge_loss_ratio')
                for name, value in logs.items():
                    if any(name.startswith(prefix) or name == prefix for prefix in cd_metric_prefixes):
                        tf.summary.scalar(name, value, step=epoch)
                self.writer.flush()


class EarlyStoppingWithCooldown(tf.keras.callbacks.EarlyStopping):
    """Early stopping with a cooldown period after patience is reached."""

    def __init__(self, cooldown: int = 0, min_epochs: int = 0, **kwargs):
        """
        Args:
            cooldown: Number of epochs to continue training after patience is met.
            min_epochs: Minimum number of epochs to train before stopping.
            **kwargs: Arguments for the standard EarlyStopping callback.
        """
        super().__init__(**kwargs)
        self.cooldown = cooldown
        self.min_epochs = min_epochs
        self.cooldown_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        """Handles early stopping logic with cooldown and minimum epochs."""
        if epoch < self.min_epochs:
            # Reset wait counter during initial epochs to prevent premature stopping
            self.wait = 0
            return  # Don't evaluate stopping condition yet

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            # During cooldown, reset patience counter to prevent immediate stop after cooldown
            self.wait = 0
            logger.debug(f"EarlyStopping cooldown: {self.cooldown_counter} epochs remaining.")
            return  # Don't stop during cooldown

        # Call the parent's on_epoch_end to check the stopping condition
        super().on_epoch_end(epoch, logs)

        # If parent decided to stop, enter cooldown instead
        if self.model.stop_training:
            if self.cooldown > 0:
                self.model.stop_training = False  # Override parent's decision
                self.cooldown_counter = self.cooldown
                self.wait = 0  # Reset wait counter when entering cooldown
                logger.info(f"Early stopping patience reached. Entering cooldown for {self.cooldown} epochs.")
            else:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}.")


# --- Factory function ---
def get_default_callbacks(config: Dict[str, Any], experiment_name: str, temp: str,
                         validation_data: Optional[tf.data.Dataset] = None) -> List[tf.keras.callbacks.Callback]:
    """
    Creates a default set of callbacks based on configuration.
    
    Args:
        config: Dictionary with configuration parameters.
        experiment_name: Name of the experiment.
        temp: Temperature identifier.
        validation_data: Optional validation dataset for charge/discharge analysis.
        
    Returns:
        List of configured Keras callbacks ready to use in model training.
    """
    callbacks = []
    base_dir = Path(config.get('base_dir', 'output'))

    # 1. TQDM Progress Bar
    callbacks.append(TQDMProgressBar(
        epochs=config.get('epochs'),
        steps_per_epoch=config.get('steps_per_epoch'),
        verbose=config.get('verbose', 1)
    ))

    # 2. Model Checkpoint
    checkpoint_dir = Path(config.get('checkpoint_dir', base_dir / 'checkpoints'))
    checkpoint_path = checkpoint_dir / f"{experiment_name}_{temp}_best.weights.h5"
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
    
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=config.get('monitor_metric', 'val_loss'),
        save_best_only=True,
        save_weights_only=True,
        mode='min' if 'loss' in config.get('monitor_metric', 'val_loss') else 'max',
        verbose=1
    ))

    # 3. ReduceLROnPlateau - Learning rate schedule
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor=config.get('monitor_metric', 'val_loss'),
        factor=config.get('lr_reduction_factor', 0.5),
        patience=max(3, config.get('patience', 10) // 3),
        min_lr=config.get('min_learning_rate', 1e-7),
        verbose=1,
        mode='min' if 'loss' in config.get('monitor_metric', 'val_loss') else 'max'
    ))

    # 4. Early Stopping with Cooldown
    if config.get('early_stopping', True):
        callbacks.append(EarlyStoppingWithCooldown(
            monitor=config.get('monitor_metric', 'val_loss'),
            patience=config.get('patience', 10),
            cooldown=config.get('early_stopping_cooldown', 5),
            min_epochs=config.get('min_epochs', 10),
            restore_best_weights=True,
            verbose=1,
            mode='min' if 'loss' in config.get('monitor_metric', 'val_loss') else 'max',
            min_delta=config.get('early_stopping_min_delta', 0.0001)
        ))

    # 5. Resource Monitoring
    callbacks.append(ResourceMonitorCallback(
        check_interval_batches=config.get('resource_check_interval', 50),
        mem_alert_threshold=config.get('memory_threshold', 0.9)
    ))

    # 6. Charge/Discharge Analysis (if validation data provided)
    tb_cd_callback = None
    if validation_data and config.get('analyze_charge_discharge', True):
        try:
            # Ensure validation data is cached for multiple iterations if not already
            if getattr(validation_data, '_dataset', None) is None and hasattr(validation_data, 'cache'):
                logger.info("Caching validation dataset for ChargeDischargeCallback")
                validation_data = validation_data.cache()
                
            cd_callback = ChargeDischargeCallback(
                validation_data=validation_data,
                current_feature_idx=config.get('current_feature_idx', 2),
                threshold=config.get('charge_threshold', 0.0),
                eval_interval=config.get('cd_eval_interval', 5)
            )
            callbacks.append(cd_callback)
            tb_cd_callback = cd_callback
        except Exception as e:
            logger.warning(f"Failed to create ChargeDischargeCallback: {e}")

    # 7. TensorBoard
    tensorboard_dir = Path(config.get('tensorboard_dir', base_dir / 'tensorboard'))
    os.makedirs(tensorboard_dir, exist_ok=True)  # Ensure directory exists
    
    tb_log_dir = tensorboard_dir / f"{experiment_name}_{temp}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    callbacks.append(CustomTensorBoard(
        log_dir=tb_log_dir,
        histogram_freq=config.get('tb_hist_freq', 1),
        profile_batch=config.get('tb_profile_batch', '5,15'),
        write_graph=False,
        update_freq='epoch',
        charge_discharge_callback=tb_cd_callback
    ))

    return callbacks