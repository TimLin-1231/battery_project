#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Temperature Training Coordinator - Battery Aging Prediction System

Orchestrates training across multiple temperatures, handling base model
training and subsequent transfer learning or fine-tuning steps.
Uses underlying BaseTrainer instances for individual temperature training.

Features:
- Coordinator pattern for multi-temperature workflows
- Efficient base model training and transfer learning
- Configurable layer freezing strategies
- Parallel execution support (ThreadPool/ProcessPool)
- Mixed precision training support
- Hardware-aware optimizations
"""
import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

# --- Custom Modules ---
from config.base_config import config as global_config
from core.logging import LoggerFactory
from core.memory import memory_manager, memory_cleanup
from data.data_provider import OptimizedDataLoader
from models.components.unified import ModelRegistry
from trainers.base_trainer import BaseTrainer
from trainers.pinn_trainer import PINNTrainer
from trainers.gan_trainer import GANTrainer
from utils.visualization import VisualizationManager

logger = LoggerFactory.get_logger("trainers.multitemp")

# Define trainer factory and results types
TrainerFactory = Callable[[Dict[str, Any]], BaseTrainer]
TrainingResult = Dict[str, Any]
AllResults = Dict[str, TrainingResult]

class MultiTempCoordinator:
    """Coordinates multi-temperature training and transfer learning with hardware optimization."""

    def __init__(self, model_type: str, config_override: Optional[Dict] = None):
        """Initialize the Multi-Temperature Coordinator.
        
        Args:
            model_type: Model type to train ('baseline', 'pinn', etc.)
            config_override: Configuration overrides
        """
        self._config = self._load_config(config_override or {})
        self._ensure_directories()
        
        # Core properties
        self.model_type = model_type
        self.experiment_name = self._config['experiment_name']
        self.base_temp = self._config['base_temp']
        self.transfer_temps = self._config['transfer_temps']
        self.all_results: AllResults = {}
        
        # Component initialization
        self.data_loader = OptimizedDataLoader(self._config)
        self.viz_manager = VisualizationManager(
            save_dir=self._config['figures_dir'],
            dpi=self._config.get('dpi', 300),
            style=self._config.get('plot_style', 'seaborn-v0_8-whitegrid')
        )
        
        # Set up mixed precision policy if enabled
        self._setup_mixed_precision()
        
        # Build initial model spec
        self._model_input_spec = None
        self._model_output_spec = None
        self._build_initial_model_spec()
        
        logger.info(f"MultiTemp Coordinator initialized: {self.experiment_name}")
        logger.info(f"  Model: {self.model_type}, Base: {self.base_temp}, Transfer: {self.transfer_temps}")
        logger.info(f"  Transfer Mode: {self._config['transfer_mode']}, Mixed Precision: {self._config.get('mixed_precision', False)}")

    def _setup_mixed_precision(self) -> None:
        """Configure mixed precision training if enabled."""
        if self._config.get('mixed_precision', False):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled (float16)")
        else:
            logger.info("Using default precision (float32)")

    def _load_config(self, config_override: Dict) -> Dict:
        """Load configuration, prioritizing overrides.
        
        Args:
            config_override: Dictionary with configuration overrides
            
        Returns:
            Complete configuration dictionary
        """
        # Default configuration
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        cfg = {
            'model_type': 'baseline',
            'experiment_name': f"mt_{timestamp}",
            'base_temp': global_config.get("training.base_temp", "25deg"),
            'transfer_temps': global_config.get("training.transfer_temps", ["5deg", "45deg"]),
            'parallel': global_config.get("hardware.parallel_training", False),
            'parallel_executor': global_config.get("hardware.parallel_executor_type", "thread"),
            'max_workers': max(1, global_config.get("hardware.max_training_workers", (os.cpu_count() or 4) // 2)),
            'transfer_mode': global_config.get("training.transfer_learning_mode", "fine_tuning"),
            'freeze_ratio': global_config.get("training.transfer_learning_freeze_ratio", 0.7),
            'transfer_lr_multiplier': global_config.get("training.transfer_lr_multiplier", 0.1),
            'reset_between_temps': global_config.get("training.reset_between_temps", True),
            'save_final_models': True,
            'mixed_precision': global_config.get("training.mixed_precision", False),
            # Inherit trainer and system defaults
            **{k: global_config.get(f"training.{k}", v) for k, v in BaseTrainer(None, {})._config.items() if k != 'base_dir'},
            **{k: global_config.get(f"system.{k}", v) for k, v in BaseTrainer(None, {})._config.items() if k.endswith('_dir')}
        }
        
        # Apply overrides
        cfg.update(config_override)
        
        # Ensure experiment name is set
        if not cfg.get('experiment_name'):
            cfg['experiment_name'] = f"{cfg['model_type']}_{cfg['base_temp']}_{timestamp}"
            
        return cfg

    def _ensure_directories(self) -> None:
        """Create all required output directories."""
        for key in ['checkpoint_dir', 'log_dir', 'tensorboard_dir', 'figures_dir']:
            Path(self._config[key]).mkdir(parents=True, exist_ok=True)

    def _get_trainer_instance(self, model: tf.keras.Model) -> BaseTrainer:
        """Create appropriate trainer instance based on model type.
        
        Args:
            model: TensorFlow model instance
            
        Returns:
            Trainer instance (BaseTrainer or subclass)
        """
        trainer_cfg = self._config.copy()
        
        if self.model_type == 'pinn':
            return PINNTrainer(model, trainer_cfg)
        elif self.model_type == 'gan':
            return GANTrainer(model, trainer_cfg)
        else:
            return BaseTrainer(model, trainer_cfg)

    def _build_initial_model_spec(self) -> None:
        """Determine model input/output specifications from sample data."""
        logger.info("Determining model specifications...")
        
        # Select a temperature to load sample data
        temp_to_load = self.base_temp or (self.transfer_temps[0] if self.transfer_temps else "25deg")
        
        try:
            # Get sample dataset batch
            sample_ds = self.data_loader.get_dataset(
                f"{self._config['data_prefix']}_{temp_to_load}", "train", batch_size=2
            )
            if sample_ds is None:
                raise ValueError(f"Could not load sample dataset for {temp_to_load}")

            # Extract input/output specs
            self._model_input_spec = sample_ds.element_spec[0]
            self._model_output_spec = sample_ds.element_spec[1]
            logger.info(f"Model specs determined - Input: {self._model_input_spec}, Output: {self._model_output_spec}")
            
        except Exception as e:
            logger.error(f"Failed to determine model specs: {e}", exc_info=True)
            raise ValueError("Cannot determine model specifications from sample data") from e

    def _create_new_model_instance(self) -> tf.keras.Model:
        """Create a fresh model instance using the registry.
        
        Returns:
            New model instance
        """
        if self._model_input_spec is None:
            raise RuntimeError("Model specifications not determined yet")

        # Extract dimensions from specs
        seq_len = self._model_input_spec.shape[1]
        n_features = self._model_input_spec.shape[2]
        
        # Handle potential nested output spec (like PINN)
        if isinstance(self._model_output_spec, (tuple, list)):
            n_outputs = self._model_output_spec[0].shape[-1]
        else:
            n_outputs = self._model_output_spec.shape[-1]

        # Build model through registry
        return ModelRegistry.build(
            self.model_type,
            seq_len=seq_len, 
            n_features=n_features, 
            n_outputs=n_outputs,
            config=self._config
        )

    @tf.function
    def _apply_transfer_freeze(self, model: tf.keras.Model) -> None:
        """Apply layer freezing based on transfer mode.
        
        Args:
            model: Model to apply freezing to
        """
        mode = self._config['transfer_mode']
        
        if mode == 'feature_extraction':
            # Freeze early layers according to freeze ratio
            freeze_ratio = self._config['freeze_ratio']
            trainable_layers = [l for l in model.layers if l.trainable_weights]
            freeze_count = int(len(trainable_layers) * freeze_ratio)
            
            logger.info(f"Feature extraction: Freezing {freeze_count}/{len(trainable_layers)} layers ({freeze_ratio:.1%})")
            
            # Apply freezing
            for i, layer in enumerate(model.layers):
                if layer.trainable_weights:
                    layer_idx = sum(1 for l in model.layers[:i] if l.trainable_weights)
                    layer.trainable = layer_idx >= freeze_count
        elif mode == 'none':
            # Reset all layers to default trainable state
            logger.info("No transfer freezing applied (mode: none)")
        else:
            # Default to fine-tuning (all layers trainable)
            logger.info(f"Fine-tuning mode: All layers trainable")
            for layer in model.layers:
                layer.trainable = True

    def _run_single_training_task(self, temp: str, is_base: bool, 
                                base_model_weights_path: Optional[str]=None) -> TrainingResult:
        """Run training for a single temperature.
        
        Args:
            temp: Temperature identifier
            is_base: Whether this is base model training
            base_model_weights_path: Path to base model weights for transfer
            
        Returns:
            Training result dictionary
        """
        task_name = f"{self.experiment_name}_{temp}"
        logger.info(f"{'Base' if is_base else 'Transfer'} training for {temp}...")
        memory_manager.log_memory_usage(f"Start Task {temp}")

        # Create model and trainer
        model = self._create_new_model_instance()
        trainer = self._get_trainer_instance(model)
        
        # Result placeholders
        history = None
        eval_results = None
        final_model_path = None

        try:
            # Handle transfer learning if applicable
            if not is_base and self._config['transfer_mode'] != 'none' and base_model_weights_path:
                if Path(base_model_weights_path).exists():
                    # Transfer learning workflow
                    logger.info(f"Transfer learning from {Path(base_model_weights_path).name} to {temp}")
                    
                    # Load weights before compilation
                    trainer.model.load_weights(base_model_weights_path)
                    
                    # Apply transfer freezing strategy
                    self._apply_transfer_freeze(trainer.model)
                    
                    # Compile with adjusted learning rate
                    transfer_lr = self._config['learning_rate'] * self._config['transfer_lr_multiplier']
                    optimizer = trainer._get_optimizer()(learning_rate=transfer_lr)
                    trainer.compile_model(optimizer=optimizer)
                else:
                    logger.warning(f"Base weights not found at {base_model_weights_path}. Training from scratch.")
                    trainer.compile_model()
            else:
                # Standard compilation
                trainer.compile_model()

            # Load datasets
            datasets = self._load_datasets(temp)
            if not datasets:
                raise ValueError(f"Could not load datasets for {temp}")
                
            train_ds, val_ds, test_ds = datasets
            
            # Execute training
            history = trainer.train(
                train_dataset=train_ds,
                val_dataset=val_ds,
                experiment_name=self.experiment_name,
                temp=temp
            )

            # Evaluate if test data available
            if test_ds:
                eval_results = trainer.evaluate(
                    test_ds, 
                    experiment_name=self.experiment_name, 
                    temp=temp
                )
            else:
                logger.warning(f"No test dataset for {temp}, skipping evaluation")

            # Save final model if configured
            if self._config['save_final_models']:
                trainer.save_model(experiment_name=self.experiment_name, temp=temp)
                final_model_path = str(Path(self._config['checkpoint_dir']) / f"{self.experiment_name}_{temp}")

        except Exception as e:
            logger.error(f"Training for {temp} failed: {e}", exc_info=True)
            return {"temp": temp, "status": "failed", "error": str(e)}
            
        finally:
            memory_manager.log_memory_usage(f"End Task {temp}")
            memory_cleanup()  # Clean up GPU memory

        return {
            "temp": temp,
            "status": "success",
            "history": history,
            "evaluation": eval_results,
            "model_path": final_model_path
        }
        
    def _load_datasets(self, temp: str) -> Optional[Tuple]:
        """Load train, validation and test datasets for a temperature.
        
        Args:
            temp: Temperature identifier
            
        Returns:
            Tuple of (train_ds, val_ds, test_ds) or None if loading fails
        """
        prefix = f"{self._config['data_prefix']}_{temp}"
        
        try:
            train_ds = self.data_loader.get_dataset(prefix, "train")
            val_ds = self.data_loader.get_dataset(prefix, "val")
            test_ds = self.data_loader.get_dataset(prefix, "test")
            
            if train_ds is None or val_ds is None:
                logger.error(f"Required datasets missing for {temp}")
                return None
                
            return train_ds, val_ds, test_ds
            
        except Exception as e:
            logger.error(f"Failed to load datasets for {temp}: {e}")
            return None

    def train_all_temps(self) -> AllResults:
        """Train models for all configured temperatures.
        
        Returns:
            Dictionary of training results by temperature
        """
        logger.info(f"Starting multi-temperature training")
        
        # 1. Train Base Model
        base_result = self._run_single_training_task(self.base_temp, is_base=True)
        self.all_results[self.base_temp] = base_result
        
        if base_result["status"] == "failed":
            logger.error("Base model training failed. Aborting multi-temp training.")
            return self.all_results

        # Get path to the best base model weights for transfer
        base_weights_path = str(Path(self._config['checkpoint_dir']) / 
                               f"{self.experiment_name}_{self.base_temp}_best.weights.h5")
        
        if not Path(base_weights_path).exists():
            logger.warning(f"Best weights for base model not found. Transfer learning may start from scratch.")
            base_weights_path = None

        # 2. Train Transfer Models
        if self.transfer_temps:
            transfer_tasks = [(temp, False, base_weights_path) for temp in self.transfer_temps]
            
            # Determine execution mode (parallel or sequential)
            use_parallel = self._config.get('parallel', False) and len(transfer_tasks) > 1
            executor_type = self._config.get('parallel_executor', 'thread').lower()
            
            if use_parallel:
                # Select executor based on configuration
                Executor = ProcessPoolExecutor if executor_type == 'process' else ThreadPoolExecutor
                executor_name = "ProcessPool" if executor_type == 'process' else "ThreadPool"
                
                logger.info(f"Starting parallel transfer learning using {executor_name} ({self._config['max_workers']} workers)")
                
                # Execute transfer tasks in parallel
                with Executor(max_workers=self._config['max_workers']) as executor:
                    futures = [executor.submit(self._run_single_training_task, temp, is_base, weights_path)
                              for temp, is_base, weights_path in transfer_tasks]
                    
                    # Process results as they complete
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Transfer Tasks"):
                        try:
                            result = future.result()
                            self.all_results[result["temp"]] = result
                        except Exception as e:
                            logger.error(f"Error in parallel transfer task: {e}", exc_info=True)
            else:
                # Sequential execution
                logger.info("Starting sequential transfer learning")
                for temp, is_base, weights_path in transfer_tasks:
                    result = self._run_single_training_task(temp, is_base, weights_path)
                    self.all_results[temp] = result

        # 3. Generate summary and visualizations
        self._generate_summary_and_plots()
        
        logger.info("Multi-temperature training completed")
        return self.all_results

    def _generate_summary_and_plots(self) -> None:
        """Generate summary reports and visualization plots."""
        logger.info("Generating summaries and plots...")
        
        performance_summary = {}
        histories = {}
        successful_temps = []

        # Collect results
        for temp, result in self.all_results.items():
            if result.get('status') == 'success':
                successful_temps.append(temp)
                if result.get('evaluation'):
                    performance_summary[temp] = result['evaluation']
                if result.get('history'):
                    histories[temp] = result['history']

        # Save overall performance summary
        if performance_summary:
            summary_path = Path(self._config['log_dir']) / f"{self.experiment_name}_performance.json"
            try:
                serializable_summary = self._make_json_serializable(performance_summary)
                with summary_path.open('w') as f:
                    json.dump(serializable_summary, f, indent=2)
                logger.info(f"Performance summary saved to: {summary_path}")
            except Exception as e:
                logger.error(f"Failed to save summary: {e}")

        # Generate visualization plots
        try:
            # Individual training histories
            for temp, history in histories.items():
                self.viz_manager.plot_training_history(
                    history,
                    title=f"{self.experiment_name} - {temp} History",
                    filename=f"{self.experiment_name}_{temp}_history.png"
                )
                
            # Temperature comparison if multiple temps
            if len(successful_temps) > 1:
                self.viz_manager.plot_temperature_comparison(
                    metrics_by_temp=performance_summary,
                    title=f"{self.experiment_name} - Performance Comparison",
                    filename=f"{self.experiment_name}_temp_comparison.png"
                )
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON serializable format.
        
        Args:
            data: Data to convert
            
        Returns:
            JSON serializable version of the data
        """
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(i) for i in data]
        elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(data)
        elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, Path):
            return str(data)
        elif pd.isna(data):
            return None
        return data