# Refactored: scripts/train.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Training Script - Battery Aging Prediction System (Optimized Orchestrator)

Handles command-line arguments, configuration loading, setup,
instantiation of necessary components (DataLoader, ModelFactory, Trainer/Coordinator),
and execution of the training workflow (standard, multi-temp, LOOCV).
"""

import sys
import time
import datetime
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

# --- Add project root to sys.path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# --- Standard Libraries & Frameworks ---
import tensorflow as tf
import logging
# Suppress TensorFlow warnings for cleaner output (optional)
# tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Custom Modules ---
try:
    from config.base_config import config as global_config, ConfigManager, ConfigValidationError
    from core.logging import LoggerFactory # Use LoggerFactory
    from core.memory import memory_manager, memory_cleanup
    from data.data_provider import OptimizedDataLoader
    # Import model registry and specific builders if needed for direct use (though registry is preferred)
    from models.components.unified import ModelRegistry
    import models.baseline # Ensure models are registered
    import models.pinn
    # import models.gan # Uncomment if GAN used
    # import models.tcn # Uncomment if TCN used
    # import models.transformer # Uncomment if Transformer used
    from trainers.multitemp_trainer import MultiTempCoordinator # Main coordinator
    from trainers.base_trainer import BaseTrainer # For single-run mode
    # Import LOOCV trainer if created separately, or handle logic here
    # from trainers.loocv_trainer import LOOCVTrainer
    from utils.visualization import VisualizationManager
    from utils.filesystem import ensure_path
    from utils.environment import validate_environment, set_random_seed # New utility module
except ImportError as e: # pragma: no cover
    print(f"FATAL ERROR: Failed to import necessary modules: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed and the project structure is correct.", file=sys.stderr)
    sys.exit(1)

# --- Global Logger ---
logger = LoggerFactory.get_logger("scripts.train")

# --- ANSI Colors ---
class Colors:
    HEADER = '\033[95m\033[1m'; INFO = '\033[94m'; SUCCESS = '\033[92m'
    WARNING = '\033[93m'; ERROR = '\033[91m\033[1m'; ENDC = '\033[0m'
    @staticmethod
    def colored(t, c): return f"{c}{t}{Colors.ENDC}"
    @staticmethod
    def header(t): return Colors.colored(t, Colors.HEADER)
    @staticmethod
    def info(t): return Colors.colored(t, Colors.INFO)
    @staticmethod
    def success(t): return Colors.colored(t, Colors.SUCCESS)
    @staticmethod
    def warning(t): return Colors.colored(t, Colors.WARNING)
    @staticmethod
    def error(t): return Colors.colored(t, Colors.ERROR)

# --- Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Battery Aging Prediction Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core Modes ---
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON/YAML configuration file to load.")
    parser.add_argument("--mode", type=str, default="multitemp", choices=["multitemp", "single", "finetune", "loocv"],
                        help="Training mode: 'multitemp' (base + transfer), 'single' (one temp), 'finetune', 'loocv'.")
    parser.add_argument("--model-type", type=str, default=global_config.get("model.type", "baseline"),
                        choices=ModelRegistry.list_available(), help="Type of model architecture to use.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Unique name for this experiment run.")

    # --- Data & Temperature ---
    parser.add_argument("--data-prefix", type=str, default=global_config.get("data.prefix", "source"), help="Prefix for dataset files (e.g., 'source').")
    parser.add_argument("--base-temp", type=str, default=global_config.get("training.base_temp", "25deg"), help="Base temperature for training/transfer.")
    parser.add_argument("--target-temp", type=str, default=None, help="Specific target temperature for 'single' or 'finetune' mode.")
    parser.add_argument("--transfer-temps", type=str, nargs='*', default=global_config.get("training.transfer_temps", []), help="List of temperatures for transfer learning in 'multitemp' mode.")

    # --- Fine-tuning / Transfer ---
    parser.add_argument("--base-model-path", type=str, default=None, help="Path to the pre-trained base model for fine-tuning.")
    parser.add_argument("--transfer-mode", type=str, default=global_config.get("training.transfer_learning_mode", "fine_tuning"),
                        choices=["feature_extraction", "fine_tuning", "none"], help="Transfer learning strategy.")
    parser.add_argument("--freeze-ratio", type=float, default=global_config.get("training.transfer_learning_freeze_ratio", 0.7), help="Ratio of layers to freeze in feature_extraction mode.")

    # --- LOOCV Specific ---
    parser.add_argument("--fold-pattern", type=str, default=global_config.get("training.fold_pattern", "fold"), help="Filename pattern to identify folds for LOOCV.")
    parser.add_argument("--fold-min", type=int, default=global_config.get("training.fold_min", 0), help="Minimum fold number for LOOCV.")
    parser.add_argument("--fold-max", type=int, default=global_config.get("training.fold_max", 9), help="Maximum fold number for LOOCV.")

    # --- Core Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per replica.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--accum-steps", type=int, default=None, help="Gradient accumulation steps.")

    # --- Features & Flags ---
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable mixed precision (FP16).")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable parallel multi-temp training.")
    parser.add_argument("--reset", action="store_true", default=False, help="Force retraining, ignore existing checkpoints/status.")
    parser.add_argument("--debug", action="store_true", default=global_config.get("system.debug", False), help="Enable debug logging.")

    # --- Resource Control ---
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers for multi-temp.")
    parser.add_argument("--seed", type=int, default=global_config.get("system.random_seed", 42), help="Random seed for reproducibility.")

    return parser.parse_args()

def setup_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads config file, merges with args, and updates global config."""
    if args.config:
        if Path(args.config).exists():
             config_manager = ConfigManager(config_path=args.config)
             logger.info(f"Loaded base configuration from: {args.config}")
             # Update global config with file contents
             global_config.update(config_manager.list_all())
        else:
             logger.error(f"Configuration file not found: {args.config}. Using defaults.")

    # Create override dict from non-None args
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}

    # Generate experiment name if not provided
    if not overrides.get('experiment_name'):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        mode_prefix = args.mode
        if args.mode == 'loocv': mode_prefix = f"loocv_{args.fold_pattern}"
        elif args.mode == 'single': mode_prefix = f"single_{args.target_temp or args.base_temp}"
        elif args.mode == 'finetune': mode_prefix = f"ft_{args.target_temp or args.base_temp}"
        overrides['experiment_name'] = f"{mode_prefix}_{args.model_type}_{timestamp}"
        logger.info(f"Generated experiment name: {overrides['experiment_name']}")

    # Update the global config instance with overrides
    # Convert relevant keys to nested structure if needed (e.g., training.epochs)
    nested_overrides = {}
    for key, value in overrides.items():
         # Map simple args to nested config structure if necessary
         # Example: args.lr -> training.learning_rate
         if key == 'lr': nested_overrides['training.learning_rate'] = value
         elif key == 'fp16': nested_overrides['training.fp16_training'] = value
         elif key == 'accum_steps': nested_overrides['training.gradient_accumulation_steps'] = value
         elif key == 'parallel': nested_overrides['hardware.parallel_training'] = value
         elif key == 'max_workers': nested_overrides['hardware.max_training_workers'] = value
         elif key == 'freeze_ratio': nested_overrides['training.transfer_learning_freeze_ratio'] = value
         elif key == 'transfer_mode': nested_overrides['training.transfer_learning_mode'] = value
         elif key in ['epochs', 'batch_size', 'patience', 'base_temp', 'transfer_temps', 'data_prefix',
                      'fold_pattern', 'fold_min', 'fold_max']:
             nested_overrides[f"training.{key}"] = value # Assume training scope for these
         elif key in ['model_type']:
              nested_overrides[f"model.{key}"] = value # Assume model scope
         elif key.endswith('_dir'):
              nested_overrides[f"system.{key}"] = str(ensure_path(value)) # Ensure paths are strings for config
         else:
              # Assume system scope or handle directly if needed
              nested_overrides[f"system.{key}"] = value # Example placement


    global_config.update(nested_overrides) # Update global config

    # Return the final effective config as a dict
    final_config = global_config.list_all()
    logger.info(f"Effective configuration loaded for experiment: {final_config.get('system.experiment_name')}")
    if args.debug: logger.debug(f"Full Effective Config: {json.dumps(final_config, indent=2, default=str)}")

    return final_config


def display_config_summary(effective_config: Dict[str, Any]):
    """Displays a summary of the effective configuration."""
    exp_name = effective_config.get('system.experiment_name', 'N/A')
    model_type = effective_config.get('model.model_type', 'N/A')
    mode = effective_config.get('system.mode', 'N/A') # Get mode from config now

    print(Colors.header("\n" + "=" * 60))
    print(Colors.header(f"     Starting Experiment: {exp_name}     "))
    print(Colors.header("=" * 60))
    print(f"Mode: {Colors.info(mode)} | Model Type: {Colors.info(model_type)}")

    if mode == 'loocv':
         print(f"Fold Pattern: {Colors.info(effective_config.get('training.fold_pattern', 'N/A'))} "
               f"({effective_config.get('training.fold_min', 'N/A')}-{effective_config.get('training.fold_max', 'N/A')})")
    elif mode == 'single':
         target_temp = effective_config.get('system.target_temp') or effective_config.get('training.base_temp')
         print(f"Target Temp: {Colors.info(target_temp)}")
    elif mode == 'finetune':
         target_temp = effective_config.get('system.target_temp') or effective_config.get('training.base_temp')
         print(f"Target Temp: {Colors.info(target_temp)}")
         print(f"Base Model: {Colors.info(effective_config.get('system.base_model_path', 'N/A'))}")
         print(f"Transfer Mode: {Colors.info(effective_config.get('training.transfer_learning_mode', 'N/A'))}")
    else: # multitemp
         print(f"Base Temp: {Colors.info(effective_config.get('training.base_temp', 'N/A'))}")
         print(f"Transfer Temps: {Colors.info(effective_config.get('training.transfer_temps', []))}")
         print(f"Transfer Mode: {Colors.info(effective_config.get('training.transfer_learning_mode', 'N/A'))}")
         print(f"Parallel Training: {Colors.success('Enabled') if effective_config.get('hardware.parallel_training', False) else Colors.warning('Disabled')}")


    print(f"Epochs: {Colors.info(effective_config.get('training.epochs', 'N/A'))} | "
          f"Batch Size: {Colors.info(effective_config.get('training.batch_size', 'N/A'))} | "
          f"Learning Rate: {Colors.info(effective_config.get('training.learning_rate', 'N/A'))}")
    print(f"Mixed Precision: {Colors.success('Enabled') if effective_config.get('training.fp16_training', False) else Colors.warning('Disabled')}")
    print(f"Output Dir: {Colors.info(effective_config.get('system.output_dir', 'N/A'))}")
    print(Colors.header("=" * 60 + "\n"))


# --- Main Orchestration ---
def main():
    """Main orchestration function."""
    args = parse_arguments()

    # Setup configuration (loads file, merges args, updates global_config)
    try:
        effective_config = setup_configuration(args)
    except ConfigValidationError as e:
         logger.error(f"Configuration Error:\n{e}")
         return 1
    except Exception as e:
         logger.error(f"Error setting up configuration: {e}", exc_info=True)
         return 1

    # Setup Logging Level based on final config
    log_level = effective_config.get('system.log_level', 'INFO')
    logger.setLevel(logging.getLevelName(log_level.upper()))
    # Update handlers' levels if needed (assuming handlers are accessible via LoggerFactory)
    # LoggerFactory.update_handler_levels(logger.level)

    # Set random seed
    set_random_seed(effective_config.get('system.seed', 42))

    # Display final config summary
    display_config_summary(effective_config)

    # Validate Environment
    if not validate_environment(): return 1

    # --- Select and Run Training Mode ---
    training_mode = effective_config.get('system.mode', 'multitemp')
    experiment_name = effective_config.get('system.experiment_name')
    model_type = effective_config.get('model.model_type')

    try:
        if training_mode == 'loocv':
            logger.info("Starting LOOCV training...")
            # Instantiate LOOCV coordinator/logic here
            # loocv_trainer = LOOCVTrainer(model_type, effective_config)
            # results = loocv_trainer.run_loocv()
            logger.error("LOOCV training mode not fully implemented in this refactoring stage.")
            # Placeholder for LOOCV execution
            results = {}

        elif training_mode == 'finetune':
            logger.info("Starting fine-tuning...")
            if not effective_config.get('system.base_model_path'):
                 logger.error("Fine-tuning requires --base-model-path to be set.")
                 return 1
            # Fine-tuning can be a mode of MultiTempCoordinator or a separate trainer
            # Assuming it's handled by MultiTempCoordinator by setting transfer_mode
            # and providing only one target temp
            ft_config = effective_config.copy()
            ft_config['transfer_temps'] = [] # No further transfers
            ft_config['base_temp'] = effective_config.get('system.target_temp') # Target temp becomes the 'base' for this run
            # Load base model *before* coordinator init? Or handle inside?
            # Let's assume MultiTempCoordinator needs adjustment to handle explicit base model path loading.
            # For now, simulate as single temp transfer
            coordinator = MultiTempCoordinator(model_type, ft_config)
            # Manually load weights and trigger fine-tune-like training
            base_model_path = effective_config['system.base_model_path']
            target_temp = ft_config['base_temp']
            coordinator._run_single_training_task(target_temp, is_base=False, base_model_weights_path=base_model_path)
            results = coordinator.all_results


        elif training_mode == 'single':
            logger.info("Starting single temperature training...")
            target_temp = effective_config.get('system.target_temp') or effective_config.get('training.base_temp')
            single_config = effective_config.copy()
            single_config['base_temp'] = target_temp # Set the only temp to train
            single_config['transfer_temps'] = [] # No transfers

            coordinator = MultiTempCoordinator(model_type, single_config)
            coordinator.train_all_temps() # Will only train the base_temp
            results = coordinator.all_results


        else: # Default to multitemp
            logger.info("Starting multi-temperature training...")
            coordinator = MultiTempCoordinator(model_type, effective_config)
            results = coordinator.train_all_temps()


        # --- Post-Training ---
        logger.info("Training process finished.")
        # Optional: Final evaluation summary print
        # print_evaluation_summary(results) # Define this helper if needed

        return 0

    except Exception as e:
        logger.critical(f"Unhandled exception during training orchestration: {e}", exc_info=True)
        return 1
    finally:
        memory_cleanup()


if __name__ == "__main__":
    # Add guard for multiprocessing safety if ProcessPoolExecutor is used
    # multiprocessing.freeze_support() # For Windows compatibility

    exit_code = main()
    logger.info(f"Script exiting with code {exit_code}.")
    sys.exit(exit_code)
