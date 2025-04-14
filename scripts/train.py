# battery_project/scripts/train.py

import argparse
import logging
import os
import sys
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any

# --- Add Ray Tune imports ---
# Wrap in try-except to make Ray Tune an optional dependency
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.integration.keras import TuneReportCallback # Keras callback for reporting
    # from ray.air import session # Newer API for reporting
    # from ray.air.integrations.keras import ReportCheckpointCallback # Newer callback
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    # Define placeholders if ray is not installed to avoid NameErrors
    ray = None
    tune = None
    ASHAScheduler = None
    TuneReportCallback = None
    logging.warning("Ray Tune library not found. HPO functionality will be disabled. "
                    "Install with: pip install 'ray[tune]'")
# ---------------------------

# Append project root to sys.path to allow relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Project specific imports (ensure correct paths)
from battery_project.core.logging import setup_logging
from battery_project.core.config import load_config, log_config, override_config_with_args, get_effective_config, Config
from battery_project.core.system import setup_system_environment, get_hardware_adapter
from battery_project.data.data_provider import OptimizedDataLoader
from battery_project.models import ModelRegistry # Access model builders
from battery_project.trainers import TrainerFactory, BaseTrainer # Access trainer factory and base class

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a battery prediction model.")
    parser.add_argument("--config", type=str, default="config/base_config.py",
                        help="Path to the base Python configuration file.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory path (processed data location).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override base output directory for results.")
    parser.add_argument("--experiment-name", type=str, default="battery_training_run",
                        help="Name for the training experiment run directory.")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model name specified in config (e.g., cnn_bilstm_attention, pinn).")
    parser.add_argument("--trainer-name", type=str, default=None,
                        help="Override trainer name specified in config (e.g., base, gan, pinn).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--load-weights", action="store_true",
                        help="Load pre-trained weights before starting training.")
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Path to weights file (.h5 or TF checkpoint) to load (used with --load-weights or --finetune-mode).")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set the logging level.")
    # --- HPO Arguments ---
    parser.add_argument("--hpo", action="store_true",
                        help="Enable Hyperparameter Optimization using Ray Tune.")
    parser.add_argument("--num-hpo-trials", type=int, default=10,
                        help="Number of HPO trials (different hyperparameter sets) to run.")
    parser.add_argument("--hpo-search-space-file", type=str, default="config/hpo_search_space.py",
                        help="Path to Python file defining the HPO search space dictionary.")
    # --- Fine-tuning Argument ---
    parser.add_argument("--finetune-mode", action="store_true",
                        help="Enable fine-tuning mode. Requires --load-weights or a valid weights_path in config.")
    parser.add_argument("--finetune-fraction", type=float, default=1.0,
                        help="Fraction of training data (0.0 to 1.0) to use when in fine-tuning mode. Default uses all data.")

    return parser.parse_args()

# --- HPO Training Function ---
def train_function_for_tune(tune_hpo_config: Dict[str, Any], base_config: Dict, args: argparse.Namespace):
    """
    The function that Ray Tune will execute for each hyperparameter trial.

    Args:
        tune_hpo_config: Dictionary of hyperparameters generated by Ray Tune for this specific trial.
        base_config: The base configuration dictionary loaded from the config file.
        args: Original parsed command-line arguments (used for non-tunable settings like paths).
    """
    # 1. Create a trial-specific configuration by merging HPO params
    # Perform a deep copy to avoid modifying the original base_config
    import copy
    trial_config_dict = copy.deepcopy(base_config)

    # Merge HPO parameters into the trial config
    # Example merging logic (needs to be adapted based on search_space keys)
    for key, value in tune_hpo_config.items():
        try:
            # Handle nested keys (e.g., "trainer.lr", "model.baseline.dropout")
            parts = key.split('.')
            d = trial_config_dict
            for part in parts[:-1]:
                d = d.setdefault(part, {}) # Create intermediate dicts if they don't exist
            d[parts[-1]] = value
            # logger.debug(f"Applied HPO param: {key} = {value}")
        except Exception as merge_err:
             # Log error but continue, HPO might provide unexpected keys
             logger.error(f"Error merging HPO key '{key}': {merge_err}", exc_info=True)


    # Convert back to Config object if needed for type consistency downstream
    # trial_config = Config(trial_config_dict)
    # Or just use the dictionary 'trial_config_dict' directly

    # 2. Setup Trial Directory and Logging
    # Ray Tune provides a unique directory for each trial's artifacts.
    try:
        # Newer Ray AIR API:
        # trial_dir = ray.train.get_context().get_trial_dir()
        # Fallback to older Tune API:
        trial_dir = tune.get_trial_dir()
        logger.info(f"Ray Tune trial directory: {trial_dir}")
    except Exception as e:
        logger.error(f"Could not get Ray Tune trial directory: {e}. Using default 'hpo_trial'.")
        trial_dir = os.path.join(args.output_dir or "results", "hpo_trial", f"trial_{time.time()}") # Fallback

    log_dir = os.path.join(trial_dir, "logs")
    weights_dir = os.path.join(trial_dir, "weights")
    ensure_dir(log_dir)
    ensure_dir(weights_dir)

    # Update config paths for this trial
    trial_config_dict['core']['log_dir'] = log_dir
    trial_config_dict['core']['output_dir'] = trial_dir # Set trial dir as base output dir
    # Ensure model weights path is within the trial directory
    model_name_trial = trial_config_dict.get("model", {}).get("name", "model")
    weights_filename = f"best_{model_name_trial}.h5" # Example filename
    trial_config_dict['model']['weights_path'] = os.path.join(weights_dir, weights_filename)
    # Set checkpoint dir for callbacks within weights_dir
    trial_config_dict['trainer']['checkpoint_dir'] = weights_dir


    # Setup logging specifically for this trial
    # Consider using Ray's logging utilities if available, or file handler per trial
    setup_logging(log_dir=log_dir, level=args.log_level, filename="trial_train.log")
    logger.info(f"--- Starting Ray Tune Trial ---")
    logger.info(f"Trial HPO Params: {tune_hpo_config}")
    # Log the fully merged config for this trial (can be verbose)
    # log_config(trial_config_dict, logger, log_level=logging.DEBUG)


    # 3. Setup System and Hardware
    # This might be redundant if setup is global, but safer per trial if isolation needed.
    setup_system_environment(trial_config_dict.get("core", {}))
    hardware_adapter = get_hardware_adapter(trial_config_dict.get("hardware", {}))

    # 4. Load Data
    # Data loading should ideally be done once and shared if possible, but loading per trial is safer.
    try:
        data_loader = OptimizedDataLoader(trial_config_dict, hardware_adapter)
        train_iter, val_iter, train_steps, val_steps = data_loader.get_train_val_iterators()
        if train_iter is None:
            raise ValueError("Failed to get training data iterator for HPO trial.")
        input_shape = data_loader.get_feature_shape()
        output_shape = data_loader.get_target_shape()
        if input_shape is None or output_shape is None:
             raise ValueError("Failed to get input/output shape from data loader.")
    except Exception as data_err:
        logger.error(f"Data loading failed for HPO trial: {data_err}", exc_info=True)
        tune.report(loss=float('inf'), error=str(data_err)) # Report failure to Tune
        return # Stop trial

    # 5. Build Model
    try:
        model_name = trial_config_dict.get("model", {}).get("name")
        # Get model specific config nested under model_name or use general model config
        model_specific_cfg = trial_config_dict.get("model", {}).get(model_name, trial_config_dict.get("model", {}))
        model = ModelRegistry.get(model_name)(
            input_shape=input_shape,
            output_shape=output_shape,
            cfg=model_specific_cfg
        )
    except Exception as model_err:
        logger.error(f"Model building failed for HPO trial (model: {model_name}): {model_err}", exc_info=True)
        tune.report(loss=float('inf'), error=str(model_err))
        return

    # 6. Instantiate Trainer
    try:
        trainer_name = trial_config_dict.get("trainer", {}).get("name")
        trainer = TrainerFactory.create_trainer(trainer_name, trial_config_dict, model, hardware_adapter)
    except Exception as trainer_err:
        logger.error(f"Trainer instantiation failed for HPO trial (trainer: {trainer_name}): {trainer_err}", exc_info=True)
        tune.report(loss=float('inf'), error=str(trainer_err))
        return

    # 7. Setup Callbacks for Ray Tune Reporting
    # Use TuneReportCallback to automatically report metrics to Ray Tune.
    # Ensure metric names match those expected by the scheduler (e.g., "val_loss").
    hpo_callbacks = []
    if TuneReportCallback:
        metrics_to_report = {
            "loss": "loss", # Train loss
            "val_loss": "val_loss" # Validation loss (primary metric for scheduler)
        }
        # Add other metrics from config if needed
        metrics_names = trial_config_dict.get("trainer", {}).get("metrics", [])
        for m_name in metrics_names:
            # Map metric name to how it appears in logs (e.g., RootMeanSquaredError -> val_root_mean_squared_error)
            # This mapping needs to be robust based on Keras metric naming conventions.
            if isinstance(m_name, str):
                log_metric_name = m_name.lower()
                # Handle potential Keras class names vs lowercase strings
                if "rmse" in log_metric_name or "rootmeansquarederror" in log_metric_name:
                    metrics_to_report["val_rmse"] = "val_root_mean_squared_error"
                elif "mae" in log_metric_name or "meanabsoluteerror" in log_metric_name:
                    metrics_to_report["val_mae"] = "val_mean_absolute_error"
                elif "r2" in log_metric_name or "r2score" in log_metric_name:
                     metrics_to_report["val_r2"] = "val_r2_score" # Adjust based on actual metric name in logs
                else: # Report other metrics directly if name matches logs
                     metrics_to_report[f"val_{log_metric_name}"] = f"val_{log_metric_name}"

        logger.debug(f"Metrics to report to Ray Tune: {metrics_to_report}")
        hpo_callbacks.append(TuneReportCallback(metrics=metrics_to_report, on="epoch_end"))
    else:
        logger.warning("TuneReportCallback not available. Metrics will not be reported to Ray Tune automatically.")

    # Combine with trainer's default callbacks (e.g., Checkpoint, EarlyStopping)
    # Ensure checkpoint paths are relative to the trial directory
    default_callbacks = trainer.get_default_callbacks(checkpoint_dir=weights_dir)
    all_callbacks = default_callbacks + hpo_callbacks


    # 8. Run Training for the Trial
    try:
        logger.info(f"Starting training for HPO trial...")
        history = trainer.train(
            train_data=train_iter,
            val_data=val_iter,
            epochs=trial_config_dict.get("trainer", {}).get("epochs"), # Use trial's epoch count
            train_steps=train_steps,
            val_steps=val_steps,
            callbacks=all_callbacks
        )
        logger.info(f"HPO trial training finished.")
        # Reporting is handled by the callback. Can report final metrics manually if needed.
        # Example: tune.report(final_val_loss=history.history['val_loss'][-1])
    except Exception as train_err:
        logger.error(f"Exception during HPO trial training: {train_err}", exc_info=True)
        # Report failure to Ray Tune
        tune.report(loss=float('inf'), error=str(train_err)) # Report infinite loss on error


# --- Main Execution Logic ---
def main(args: argparse.Namespace):
    """Main function to orchestrate training or HPO."""
    # 1. Load and Process Configuration
    config_dict = load_config(args.config)
    # Override with command-line args (excluding HPO args if in HPO mode)
    config_dict = override_config_with_args(config_dict, args)
    effective_config_dict = get_effective_config(config_dict)

    # Determine output directories for normal run
    output_dir = effective_config_dict.get("core", {}).get("output_dir", "results")
    experiment_dir = os.path.join(output_dir, args.experiment_name)
    log_dir_main = effective_config_dict.get("core", {}).get("log_dir", os.path.join(experiment_dir, "logs"))
    weights_dir_main = os.path.join(experiment_dir, "weights") # Default weights dir


    # --- HPO Execution Path ---
    if args.hpo:
        if not RAY_AVAILABLE:
            logger.error("Ray Tune is not installed, cannot perform HPO. Install with: pip install 'ray[tune]'")
            sys.exit(1)

        # Setup logging for the main HPO process (driver)
        setup_logging(log_dir=log_dir_main, level=args.log_level, filename="hpo_driver.log")
        ensure_dir(log_dir_main)
        logger.info("=" * 50)
        logger.info("Starting Hyperparameter Optimization with Ray Tune...")
        logger.info("=" * 50)
        log_config(effective_config_dict, logger) # Log base config for HPO run

        # Load search space from the specified Python file
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("hpo_search_space_module", args.hpo_search_space_file)
            if spec is None or spec.loader is None:
                 raise ImportError(f"Could not load spec for HPO config: {args.hpo_search_space_file}")
            hpo_module = importlib.util.module_from_spec(spec)
            sys.modules["hpo_search_space_module"] = hpo_module # Add to sys modules temporarily
            spec.loader.exec_module(hpo_module)
            if not hasattr(hpo_module, 'search_space'):
                 raise AttributeError(f"HPO config file {args.hpo_search_space_file} must define a 'search_space' dictionary.")
            search_space = hpo_module.search_space
            logger.info(f"Loaded HPO search space from {args.hpo_search_space_file}: {search_space}")
        except Exception as e:
            logger.error(f"Failed to load HPO search space from {args.hpo_search_space_file}: {e}", exc_info=True)
            sys.exit(1)

        # Configure Scheduler (e.g., ASHAScheduler)
        # Metric should match a primary metric reported by TuneReportCallback (e.g., 'val_loss')
        scheduler = None
        if ASHAScheduler:
             scheduler = ASHAScheduler(
                 metric="val_loss", # Primary metric to monitor
                 mode="min",         # Minimize validation loss
                 max_t=effective_config_dict.get("trainer", {}).get("epochs", 100), # Max epochs per trial
                 grace_period=effective_config_dict.get("trainer", {}).get("hpo_grace_period", 10), # Min epochs before stopping
                 reduction_factor=effective_config_dict.get("trainer", {}).get("hpo_reduction_factor", 2)
             )
             logger.info(f"Using ASHAScheduler (metric=val_loss, mode=min).")
        else:
             logger.warning("ASHAScheduler not available. Running HPO without advanced scheduling.")


        # Configure resources per trial (adjust based on hardware)
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        resources_per_trial = {"cpu": 2, "gpu": 1 if num_gpus > 0 else 0}
        logger.info(f"Ray Tune resources per trial: {resources_per_trial}")

        # Initialize Ray (if not already running in a cluster)
        if not ray.is_initialized():
             try:
                 # Try initializing Ray with available resources
                 ray.init(
                     # num_cpus=os.cpu_count(), # Let Ray detect CPUs
                     # num_gpus=num_gpus, # Let Ray detect GPUs
                     configure_logging=False, # Avoid Ray overriding root logger
                     log_to_driver=True # Send trial logs to driver process output
                 )
             except Exception as ray_init_err:
                 logger.error(f"Ray initialization failed: {ray_init_err}. Check Ray setup.", exc_info=True)
                 sys.exit(1)

        # Define where Ray Tune stores its results locally
        local_dir = os.path.join(output_dir, "ray_tune_results")
        ensure_dir(local_dir)

        # Run the Tune experiment
        logger.info(f"Starting tune.run with {args.num_hpo_trials} trials...")
        analysis = tune.run(
            tune.with_parameters(train_function_for_tune, base_config=effective_config_dict, args=args),
            resources_per_trial=resources_per_trial,
            config=search_space, # The hyperparameter search space
            num_samples=args.num_hpo_trials, # Number of trials to run
            scheduler=scheduler,             # The trial scheduler (optional)
            name=f"{args.experiment_name}_hpo", # Experiment name in Ray Tune dashboard/results
            local_dir=local_dir,             # Directory to store trial results
            # progress_reporter=tune.CLIReporter(metric_columns=["loss", "val_loss", "val_rmse"]), # Optional CLI reporter
            keep_checkpoints_num=1,            # Keep only the best checkpoint per trial
            checkpoint_score_attr="min-val_loss", # Select best checkpoint based on min val_loss
            verbose=1 # Set verbosity level for Tune (0=silent, 1=results, 2=trial info, 3=debug)
        )

        logger.info("=" * 50)
        logger.info("HPO finished.")
        logger.info("=" * 50)
        # Log best trial results
        best_trial = None
        try:
             best_trial = analysis.get_best_trial(
                 metric="val_loss", # Metric used for selection
                 mode="min",        # Selection mode (min or max)
                 scope="last"       # Consider the last reported value for the metric
             )
        except Exception as best_trial_err:
             logger.warning(f"Could not retrieve best trial using get_best_trial: {best_trial_err}")
             # Fallback: try finding best logdir if method above fails
             try:
                 best_logdir = analysis.get_best_logdir(metric='val_loss', mode='min')
                 logger.info(f"(Fallback) Best trial log directory: {best_logdir}")
                 # Could try loading config/results from this directory manually
             except Exception as best_logdir_err:
                   logger.error(f"Could not retrieve best trial information: {best_logdir_err}")


        if best_trial:
             logger.info(f"Best trial config found: {best_trial.config}")
             logger.info(f"Best trial final validation loss: {best_trial.last_result.get('val_loss', 'N/A'):.4f}")
             # Retrieve checkpoint path if available
             # best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="val_loss", mode="min") # Newer API
             # if best_checkpoint:
             #      logger.info(f"Best trial checkpoint path: {best_checkpoint.path}")
             # else:
             #      logger.info(f"Best trial checkpoint path (from logdir): {best_trial.logdir}") # Older API fallback
             logger.info(f"Best trial log directory: {best_trial.logdir}") # Logdir usually contains checkpoints
        else:
             logger.warning("No successful HPO trial found or best trial could not be determined.")

        # Shutdown Ray
        ray.shutdown()

    # --- Normal or Fine-tuning Execution Path ---
    else:
        # Setup logging, system, hardware adapter for a single run
        ensure_dir(log_dir_main)
        ensure_dir(weights_dir_main)
        setup_logging(log_dir=log_dir_main, level=args.log_level, filename="train.log")
        logger.info("=" * 50)
        run_mode = "Fine-tuning" if args.finetune_mode else "Training"
        logger.info(f"Starting {run_mode} Run: {args.experiment_name}")
        logger.info("=" * 50)
        log_config(effective_config_dict, logger)

        setup_system_environment(effective_config_dict.get("core", {}))
        hardware_adapter = get_hardware_adapter(effective_config_dict.get("hardware", {}))

        # Data Loading
        try:
             data_loader = OptimizedDataLoader(effective_config_dict, hardware_adapter)
             train_iter, val_iter, train_steps, val_steps = data_loader.get_train_val_iterators()
             if train_iter is None:
                  raise ValueError("Failed to get training data iterator.")
             input_shape = data_loader.get_feature_shape()
             output_shape = data_loader.get_target_shape()
             if input_shape is None or output_shape is None:
                  raise ValueError("Failed to get input/output shape from data loader.")
        except Exception as data_err:
             logger.error(f"Data loading failed: {data_err}", exc_info=True)
             sys.exit(1)


        # --- Handle Fine-tuning Data Fraction ---
        is_finetuning = args.finetune_mode
        if is_finetuning and args.finetune_fraction < 1.0 and args.finetune_fraction > 0.0:
            if train_steps is None:
                 logger.error("Cannot apply fine-tune fraction: Unknown number of training steps.")
                 # Optionally, could iterate once to count, but might be slow. Exit for now.
                 sys.exit(1)
            else:
                 original_train_steps = train_steps
                 finetune_steps = max(1, int(original_train_steps * args.finetune_fraction))
                 logger.info(f"Fine-tuning mode: Using {args.finetune_fraction*100:.1f}% of training data ({finetune_steps} steps).")
                 # Use tf.data.Dataset.take() to limit the number of steps for the training iterator
                 train_iter = train_iter.take(finetune_steps)
                 train_steps = finetune_steps # Update steps for progress bar/logging
        elif is_finetuning:
             logger.info("Fine-tuning mode: Using 100% of the provided training data.")
        # ----------------------------------------

        # Model Building
        try:
             model_name = effective_config_dict.get("model", {}).get("name")
             model_specific_cfg = effective_config_dict.get("model", {}).get(model_name, effective_config_dict.get("model", {}))
             model = ModelRegistry.get(model_name)(
                 input_shape=input_shape,
                 output_shape=output_shape,
                 cfg=model_specific_cfg
             )
        except Exception as model_err:
             logger.error(f"Model building failed (model: {model_name}): {model_err}", exc_info=True)
             sys.exit(1)

        # Trainer Instantiation
        try:
             trainer_name = effective_config_dict.get("trainer", {}).get("name")
             # Ensure weights path in config points to the main run's weights dir for callbacks
             effective_config_dict['trainer']['checkpoint_dir'] = weights_dir_main
             effective_config_dict['model']['weights_path'] = os.path.join(weights_dir_main, f"best_{model_name}.h5")

             trainer = TrainerFactory.create_trainer(trainer_name, effective_config_dict, model, hardware_adapter)
        except Exception as trainer_err:
             logger.error(f"Trainer instantiation failed (trainer: {trainer_name}): {trainer_err}", exc_info=True)
             sys.exit(1)


        # Load weights if specified (for continued training or fine-tuning)
        # Fine-tuning mode REQUIRES loading weights.
        load_weights_flag = args.load_weights or is_finetuning
        # Use weights path from args first, then config.
        weights_path = args.weights_path or effective_config_dict.get("model", {}).get("weights_path")

        if load_weights_flag:
            if weights_path and os.path.exists(weights_path):
                 logger.info(f"Loading weights from: {weights_path}")
                 try:
                     # Use load_weights directly on the model instance within the trainer
                     trainer.model.load_weights(weights_path).expect_partial() # Allow partial loads
                     logger.info(f"Weights loaded successfully into model '{trainer.model.name}'.")
                 except Exception as e:
                      logger.error(f"Failed to load weights from {weights_path}: {e}", exc_info=True)
                      # Decide whether to exit or continue training from scratch
                      if is_finetuning:
                           logger.error("Cannot proceed with fine-tuning without valid weights. Exiting.")
                           sys.exit(1)
                      else:
                           logger.warning("Proceeding with training from scratch.")
            elif is_finetuning:
                 logger.error(f"Fine-tuning mode enabled but weights path not found or not provided: {weights_path}. Exiting.")
                 sys.exit(1)
            else: # load_weights was true, but path invalid and not fine-tuning
                 logger.warning(f"--load-weights specified but path not found or not provided: {weights_path}. Training from scratch.")


        # Run Training / Fine-tuning
        logger.info(f"Starting model {run_mode.lower()}...")
        try:
             trainer.train(
                 train_data=train_iter,
                 val_data=val_iter,
                 train_steps=train_steps, # Pass steps for progress bar accuracy
                 val_steps=val_steps
                 # Callbacks are handled internally by the trainer's get_default_callbacks
             )
             logger.info(f"{run_mode} finished.")
        except Exception as train_err:
             logger.error(f"An error occurred during {run_mode.lower()}: {train_err}", exc_info=True)
             sys.exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)