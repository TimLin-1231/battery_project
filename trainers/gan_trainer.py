#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAN Trainer Module - Battery Aging Prediction System

Handles training for Wasserstein GAN with Gradient Penalty (WGAN-GP) models.
Optimized for mixed precision, memory efficiency, and hardware utilization.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

# --- Custom Modules ---
from config.base_config import config as global_config
from core.logging import LoggerFactory
from core.memory import memory_manager
from trainers.callbacks import get_default_callbacks
from models.components.unified import ModelRegistry

logger = LoggerFactory.get_logger("trainers.gan")

class GradientHandler:
    """Manages gradient computation, accumulation and application for GANs.
    
    Optimized replacement for manual gradient accumulation, compatible with
    mixed precision training and pre-computed gradients.
    """
    
    def __init__(self, 
                 optimizer: tf.keras.optimizers.Optimizer,
                 accumulation_steps: int = 1,
                 name: str = "grad_handler"):
        """Initialize gradient handler with optimizer and accumulation config.
        
        Args:
            optimizer: TensorFlow optimizer to apply gradients
            accumulation_steps: Number of steps to accumulate before applying
            name: Name prefix for internal variables
        """
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self._step_count = tf.Variable(0, dtype=tf.int32, trainable=False, name=f"{name}_step")
        self._accumulators = None
        self._initialized = False
        
    def _initialize(self, var_list: List[tf.Variable]) -> None:
        """Initialize gradient accumulators for the given variables.
        
        Args:
            var_list: List of model variables to track
        """
        self._accumulators = [
            tf.Variable(tf.zeros_like(var), trainable=False) 
            for var in var_list
        ]
        self._initialized = True
    
    def accumulate_gradients(self, 
                             gradients: List[Optional[tf.Tensor]], 
                             var_list: List[tf.Variable]) -> tf.Tensor:
        """Accumulate gradients and apply when needed.
        
        Args:
            gradients: Pre-computed gradients to accumulate
            var_list: Variables corresponding to gradients
            
        Returns:
            Boolean tensor indicating if gradients were applied
        """
        if not self._initialized:
            self._initialize(var_list)
            
        # Add computed gradients to accumulators
        for i, grad in enumerate(gradients):
            if grad is not None:
                self._accumulators[i].assign_add(grad)
        
        # Increment step counter
        self._step_count.assign_add(1)
        
        # Check if we should apply accumulated gradients
        should_apply = tf.equal(self._step_count % self.accumulation_steps, 0)
        
        # Apply gradients when counter reaches accumulation steps
        def apply_grads():
            # Calculate average gradients
            avg_grads = [
                acc / tf.cast(self.accumulation_steps, acc.dtype) 
                for acc in self._accumulators
            ]
            
            # Apply non-None gradients
            grad_vars = [(g, v) for g, v in zip(avg_grads, var_list) if g is not None]
            if grad_vars:
                self.optimizer.apply_gradients(grad_vars)
                
            # Reset accumulators
            for acc in self._accumulators:
                acc.assign(tf.zeros_like(acc))
            
            # Reset counter
            self._step_count.assign(0)
            
            return tf.constant(True)
        
        # Apply or skip based on counter
        applied = tf.cond(should_apply, apply_grads, lambda: tf.constant(False))
        
        return applied


class GANTrainer:
    """Trains a Wasserstein GAN with Gradient Penalty (WGAN-GP).
    
    Optimized for mixed precision, gradient accumulation, and tensorboard monitoring.
    """

    def __init__(self,
                 generator: tf.keras.Model,
                 discriminator: tf.keras.Model,
                 config: Optional[Dict[str, Any]] = None,
                 checkpoint_dir: Optional[Union[str, Path]] = None):
        """Initialize the GANTrainer with models and configuration.

        Args:
            generator: Generator neural network model
            discriminator: Discriminator neural network model
            config: Optional config overrides for training (default: None)
            checkpoint_dir: Optional path to save checkpoints (default: None)
        """
        # Core components
        self.generator = generator
        self.discriminator = discriminator
        self._config = self._load_config(config)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Extract key config values
        self.noise_dim = self._config['noise_dim']
        self.n_critic = self._config['n_critic']
        self.gp_weight = self._config['gp_weight']
        
        # Create optimizers and gradient handlers
        self._setup_training_components()
        
        # Checkpoint and state tracking
        self._setup_checkpoints()

    def _load_config(self, config_override: Optional[Dict] = None) -> Dict[str, Any]:
        """Load GAN configuration with defaults and overrides.
        
        Args:
            config_override: Optional config values to override defaults
            
        Returns:
            Dictionary with complete configuration
        """
        # Default configuration with global fallbacks
        cfg = {
            # Model parameters
            'noise_dim': 100,
            'n_critic': 5,
            'gp_weight': 10.0,
            'g_smoothness_weight': 0.0,
            
            # Optimization parameters
            'g_lr': global_config.get("training.learning_rate", 1e-4),
            'd_lr': global_config.get("training.learning_rate", 1e-4) * 0.5,
            'beta1': global_config.get("training.beta1", 0.5),
            'g_accum_steps': global_config.get("training.gradient_accumulation_steps", 1),
            'd_accum_steps': global_config.get("training.gradient_accumulation_steps", 1),
            'clipnorm': global_config.get("training.clipnorm", 1.0),
            
            # Training parameters
            'epochs': global_config.get("training.epochs", 50),
            'use_mixed_precision': global_config.get("training.fp16_training", False),
            
            # System parameters
            'log_dir': global_config.get("system.log_dir", "logs/gan"),
            'tensorboard_dir': global_config.get("system.tensorboard_dir", "tensorboard/gan"),
        }
        
        # Apply overrides if provided
        if config_override:
            cfg.update(config_override)
            
        return cfg

    def _setup_training_components(self) -> None:
        """Setup optimizers and gradient handlers."""
        # Create optimizers with mixed precision if configured
        g_opt = tf.keras.optimizers.Adam(
            learning_rate=self._config['g_lr'], 
            beta_1=self._config['beta1'],
            clipnorm=self._config['clipnorm']
        )
        
        d_opt = tf.keras.optimizers.Adam(
            learning_rate=self._config['d_lr'], 
            beta_1=self._config['beta1'],
            clipnorm=self._config['clipnorm']
        )
        
        # Apply mixed precision wrapper if needed
        if self._config['use_mixed_precision']:
             g_opt = tf.keras.mixed_precision.LossScaleOptimizer(g_opt)
             d_opt = tf.keras.mixed_precision.LossScaleOptimizer(d_opt)
             
        self.g_optimizer = g_opt
        self.d_optimizer = d_opt
        
        # Create gradient handlers for accumulation
        self.g_grad_handler = GradientHandler(
            self.g_optimizer, 
            self._config['g_accum_steps'],
            name="gen"
        )
        
        self.d_grad_handler = GradientHandler(
            self.d_optimizer, 
            self._config['d_accum_steps'],
            name="disc"
        )

    def _setup_checkpoints(self) -> None:
        """Initialize checkpoint tracking and restore if available."""
        # Variables to track in checkpoints
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.best_g_loss = tf.Variable(float('inf'), trainable=False)
        
        # Skip if no checkpoint directory
        if not self.checkpoint_dir:
            self.checkpoint = None
            self.checkpoint_manager = None
            return
            
        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint with tracked objects
        self.checkpoint = tf.train.Checkpoint(
            step=self.step,
            epoch=self.epoch,
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            best_g_loss=self.best_g_loss
        )
        
        # Create checkpoint manager
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=3
        )
        
        # Restore latest checkpoint if available
        if self.checkpoint_manager.latest_checkpoint:
            status = self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            status.assert_existing_objects_matched()
            logger.info(f"Restored GAN checkpoint: {self.checkpoint_manager.latest_checkpoint}")
            logger.info(f"  Step: {self.step.numpy()}, Epoch: {self.epoch.numpy()}, "
                        f"Best G Loss: {self.best_g_loss.numpy():.4f}")
        else:
            logger.info("No GAN checkpoint found, starting from scratch.")

    @tf.function
    def _get_scaled_grads(self, 
                           tape: tf.GradientTape, 
                           loss: tf.Tensor, 
                           var_list: List[tf.Variable],
                           optimizer: tf.keras.optimizers.Optimizer) -> Tuple[List[tf.Tensor], tf.Tensor]:
        """Calculate gradients with mixed precision support.
        
        Args:
            tape: Gradient tape with recorded operations
            loss: Loss value to differentiate
            var_list: Variables to calculate gradients for
            optimizer: Optimizer that might wrap loss scaling
            
        Returns:
            Tuple of (unscaled gradients, original loss)
        """
        # Handle mixed precision case
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            # Get scaled loss and gradients
            scaled_loss = optimizer.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_loss, var_list)
            # Unscale gradients to original scale
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            # Standard case - just get gradients directly
            grads = tape.gradient(loss, var_list)
            
        return grads, loss

    @tf.function
    def _gradient_penalty(self, 
                          real_samples: tf.Tensor, 
                          fake_samples: tf.Tensor, 
                          conditions: tf.Tensor) -> tf.Tensor:
        """Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Batch of real data samples
            fake_samples: Batch of generated samples
            conditions: Conditional inputs for the discriminator
            
        Returns:
            Gradient penalty loss term
        """
        batch_size = tf.shape(real_samples)[0]
        
        # Generate random interpolation factors
        alpha = tf.random.uniform([batch_size, 1, 1])
        
        # Create interpolated samples between real and fake
        interpolated = alpha * real_samples + (1.0 - alpha) * fake_samples
        
        # Calculate gradients w.r.t. interpolated samples
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interp_output = self.discriminator([interpolated, conditions], training=True)
        
        # Get gradients and calculate their norm
        grads = gp_tape.gradient(interp_output, interpolated)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-8)
        
        # Calculate penalty: (||grad|| - 1)²
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.0))
        
        return gradient_penalty

    @tf.function
    def _train_discriminator(self, 
                             real_samples: tf.Tensor, 
                             conditions: tf.Tensor, 
                             noise: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Execute single discriminator training step.
        
        Args:
            real_samples: Batch of real data
            conditions: Conditional inputs
            noise: Random noise for generator
            
        Returns:
            Tuple of (discriminator loss, gradient penalty)
        """
        with tf.GradientTape() as tape:
            # Generate fake samples (no gradients needed for generator here)
            fake_samples = self.generator([noise, conditions], training=False)
            
            # Get discriminator outputs for real and fake samples
            real_output = self.discriminator([real_samples, conditions], training=True)
            fake_output = self.discriminator([fake_samples, conditions], training=True)
            
            # WGAN loss: E[D(fake)] - E[D(real)]
            d_loss_wgan = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            
            # Add gradient penalty
            gp = self._gradient_penalty(real_samples, fake_samples, conditions)
            
            # Total discriminator loss
            d_loss = d_loss_wgan + self.gp_weight * gp
        
        # Get discriminator variables
        d_vars = self.discriminator.trainable_variables
        
        # Calculate, scale, and accumulate gradients
        gradients, _ = self._get_scaled_grads(tape, d_loss, d_vars, self.d_optimizer)
        self.d_grad_handler.accumulate_gradients(gradients, d_vars)
        
        return d_loss, gp

    @tf.function
    def _train_generator(self, 
                         conditions: tf.Tensor, 
                         noise: tf.Tensor) -> tf.Tensor:
        """Execute single generator training step.
        
        Args:
            conditions: Conditional inputs
            noise: Random noise input
            
        Returns:
            Generator loss value
        """
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_samples = self.generator([noise, conditions], training=True)
            
            # Get discriminator output (no gradients for discriminator)
            fake_output = self.discriminator([fake_samples, conditions], training=False)
            
            # Generator loss: -E[D(fake)]
            g_loss = -tf.reduce_mean(fake_output)
            
            # Optional smoothness penalty
            if self._config.get("g_smoothness_weight", 0.0) > 0.0:
                smoothness = tf.reduce_mean(tf.square(fake_samples[:, 1:] - fake_samples[:, :-1]))
                g_loss += self._config["g_smoothness_weight"] * smoothness
        
        # Get generator variables
        g_vars = self.generator.trainable_variables
        
        # Calculate, scale, and accumulate gradients
        gradients, _ = self._get_scaled_grads(tape, g_loss, g_vars, self.g_optimizer)
        self.g_grad_handler.accumulate_gradients(gradients, g_vars)
        
        return g_loss

    def _run_epoch(self, 
                   dataset: tf.data.Dataset,
                   steps: int,
                   callbacks: List,
                   epoch_num: int) -> Dict[str, float]:
        """Run a single training epoch.
        
        Args:
            dataset: Dataset to iterate
            steps: Number of steps per epoch
            callbacks: Training callbacks
            epoch_num: Current epoch number
            
        Returns:
            Dictionary with epoch metrics
        """
        # Initialize metric trackers
        d_losses, g_losses, gp_values = [], [], []
        
        # Create progress bar
        pbar = tqdm(total=steps, desc=f"Epoch {epoch_num + 1}", unit="step")
        
        # Iterate over dataset for specified steps
        for step_in_epoch, (conditions, real_samples) in enumerate(dataset.take(steps)):
            batch_size = tf.shape(conditions)[0]
            noise = tf.random.normal([batch_size, self.noise_dim])
            
            # Train discriminator n_critic times
            d_loss_sum = 0.0
            gp_sum = 0.0
            for _ in range(self.n_critic):
                d_loss, gp = self._train_discriminator(real_samples, conditions, noise)
                d_loss_sum += d_loss
                gp_sum += gp
                
            # Average discriminator metrics over critic steps
            d_loss_avg = d_loss_sum / self.n_critic
            gp_avg = gp_sum / self.n_critic
            
            # Train generator once
            g_loss = self._train_generator(conditions, noise)
            
            # Store losses for epoch average
            d_losses.append(d_loss_avg.numpy())
            g_losses.append(g_loss.numpy())
            gp_values.append(gp_avg.numpy())
            
            # Increment global step
            self.step.assign_add(1)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "D Loss": f"{d_loss_avg.numpy():.4f}",
                "G Loss": f"{g_loss.numpy():.4f}",
                "GP": f"{gp_avg.numpy():.4f}"
            })
            
            # Trigger batch end callbacks
            batch_logs = {
                'd_loss': d_loss_avg.numpy(),
                'g_loss': g_loss.numpy(),
                'gp': gp_avg.numpy()
            }
            for cb in callbacks:
                cb.on_train_batch_end(step_in_epoch, batch_logs)
        
        # Close progress bar
        pbar.close()
        
        # Calculate epoch metrics
        epoch_metrics = {
            'd_loss': np.mean(d_losses),
            'g_loss': np.mean(g_losses),
            'gp': np.mean(gp_values)
        }
        
        return epoch_metrics

    def train(self, 
              dataset: tf.data.Dataset, 
              epochs: Optional[int] = None,
              steps_per_epoch: Optional[int] = None, 
              callbacks: Optional[List] = None) -> Dict[str, List[float]]:
        """Train the GAN for specified epochs.
        
        Args:
            dataset: Training dataset (batched)
            epochs: Number of epochs to train (default: from config)
            steps_per_epoch: Steps per epoch (default: dataset size)
            callbacks: List of callbacks (default: [])
            
        Returns:
            Dictionary with training history
        """
        # Set epochs from config if not specified
        epochs = epochs if epochs is not None else self._config['epochs']
        initial_epoch = self.epoch.numpy()
        
        logger.info(f"Starting GAN training from epoch {initial_epoch + 1} for {epochs} epochs")
        
        # Determine steps per epoch if not provided
        if steps_per_epoch is None:
            cardinality = tf.data.experimental.cardinality(dataset).numpy()
            if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
                steps_per_epoch = 1000  # Default if unknown
                logger.warning(f"Dataset size unknown, using default steps_per_epoch={steps_per_epoch}")
            elif cardinality == tf.data.experimental.INFINITE_CARDINALITY:
                logger.error("Dataset must not be infinite for GAN training")
                return {}
            else:
                steps_per_epoch = cardinality
                
        # Initialize callbacks
        if callbacks is None:
            callbacks = []
            
        # Set model for callbacks
        for cb in callbacks:
            cb.set_model(self.generator)
            
        # Trigger train begin
        for cb in callbacks:
            cb.on_train_begin()
            
        # Track training history
        history = {'d_loss': [], 'g_loss': [], 'gp': []}
        
        # Main training loop
        for epoch_num in range(initial_epoch, initial_epoch + epochs):
            # Update epoch counter
            self.epoch.assign(epoch_num)
            
            # Time the epoch
            start_time = tf.timestamp()
            
            # Run epoch and get metrics
            metrics = self._run_epoch(dataset, steps_per_epoch, callbacks, epoch_num)
            
            # Calculate epoch duration
            epoch_time = tf.timestamp() - start_time
            
            # Log epoch results
            logger.info(f"Epoch {epoch_num + 1} Summary - "
                      f"Time: {epoch_time.numpy():.2f}s, "
                      f"D Loss: {metrics['d_loss']:.4f}, "
                      f"G Loss: {metrics['g_loss']:.4f}, "
                      f"GP: {metrics['gp']:.4f}")
            
            # Update history
            for k, v in metrics.items():
                history[k].append(v)
            
            # Trigger epoch end callbacks
            for cb in callbacks:
                cb.on_epoch_end(epoch_num, metrics)
            
            # Save checkpoint if improved
            if self.checkpoint_manager and metrics['g_loss'] < self.best_g_loss:
                self.best_g_loss.assign(metrics['g_loss'])
                save_path = self.checkpoint_manager.save()
                logger.info(f"Checkpoint saved: {save_path} (G Loss: {metrics['g_loss']:.4f})")
                
        # Trigger train end callbacks
        for cb in callbacks:
            cb.on_train_end()
            
        logger.info("GAN Training Finished")
        
        return history