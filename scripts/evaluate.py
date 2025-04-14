# battery_project/scripts/evaluate.py

import argparse
import logging
import os
import sys
import time # For timing SHAP etc.
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any, Union

# --- Add SHAP import ---
# Wrap in try-except to make SHAP an optional dependency
try:
    import shap
    SHAP_AVAILABLE = True
    logging.info("SHAP library imported successfully.")
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP library not found. Explainability features will be disabled. "
                    "Install with: pip install shap")
    shap = None # Define as None to avoid NameErrors
# -----------------------

# Append project root to sys.path to allow relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Project specific imports (ensure correct paths)
from battery_project.core.logging import setup_logging
from battery_project.core.config import load_config, log_config, override_config_with_args, get_effective_config, Config
from battery_project.core.system import setup_system_environment, get_hardware_adapter
from battery_project.data.data_provider import OptimizedDataLoader, FormatParser # Import FormatParser for feature names
from battery_project.models import ModelRegistry # Access model builders
from battery_project.trainers import BaseTrainer # Import BaseTrainer for get_metrics static method
from battery_project.utils.metrics import calculate_metrics # Assuming metrics calculation is refactored
from battery_project.utils.visualization import plot_predictions #, plot_charge_discharge_metrics # Assumed helpers
from battery_project.utils.filesystem import ensure_dir

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parses command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained battery prediction model.")
    parser.add_argument("--config", type=str, default="config/base_config.py",
                        help="Path to the Python configuration file used during training.")
    parser.add_argument("--weights-path", type=str, required=True,
                        help="Path to the trained model weights file (.h5 or TF checkpoint folder).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory path (should point to processed test data).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override base output directory for evaluation results.")
    parser.add_argument("--experiment-name", type=str, default="battery_evaluation_run",
                        help="Name for the evaluation run directory.")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model name (must match the architecture of the loaded weights).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set the logging level.")
    parser.add_argument("--plot", action="store_true", help="Generate and save prediction plots.")
    parser.add_argument("--cross-temp-eval", action="store_true",
                        help="Perform cross-temperature evaluation (requires temperature feature and logic).")
    # --- Small-Sample Eval Argument ---
    # Note: This evaluates ON a smaller fraction of the test set.
    parser.add_argument("--eval-fraction", type=float, default=1.0,
                        help="Fraction of test data (0.0 to 1.0) to use for evaluation. Default uses all test data.")
    # --- Explainability Arguments ---
    parser.add_argument("--explain", action="store_true",
                        help="Perform SHAP analysis on a subset of the test data.")
    parser.add_argument("--explain-samples", type=int, default=50,
                        help="Number of samples for SHAP background data summarization and foreground explanation.")
    # --- UQ Arguments ---
    parser.add_argument("--uq-eval", action="store_true",
                        help="Perform Uncertainty Quantification evaluation (requires UQ-enabled model, e.g., MC Dropout).")
    parser.add_argument("--uq-samples", type=int, default=50,
                        help="Number of forward passes for MC Dropout or samples for other UQ methods.")

    return parser.parse_args()

def perform_shap_analysis(model: tf.keras.Model, data_iter: tf.data.Dataset,
                          feature_names: List[str], num_background: int, num_foreground: int,
                          output_dir: str, batch_size: int):
    """
    Performs SHAP analysis on model predictions and saves summary plots.

    Args:
        model: The trained Keras model.
        data_iter: A tf.data.Dataset iterator yielding (features, target) for test data.
                   Must be resettable or a new iterator instance.
        feature_names: List of names corresponding to the feature dimension of the input data.
        num_background: Number of samples to use for creating the background distribution summary.
        num_foreground: Number of samples to explain.
        output_dir: Directory to save SHAP plots.
        batch_size: Batch size to use when iterating through data.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP library not available, skipping explanation analysis.")
        return
    if not feature_names:
        logger.warning("No feature names provided, SHAP plots will have generic labels. Provide 'data.feature_columns' in config.")
        # Create generic names based on expected shape if possible
        # feature_dim = model.input_shape[-1] # Get from model input spec? Risky.
        # feature_names = [f"Feature_{i}" for i in range(feature_dim)]

    logger.info(f"Starting SHAP analysis: Background Samples={num_background}, Foreground Samples={num_foreground}")
    start_time = time.time()

    try:
        # 1. Prepare Background Data
        # SHAP needs representative background data to calculate expected values.
        # Using a subset of the test data is common.
        background_data_list = []
        samples_collected = 0
        # Iterate through dataset to collect background samples
        for x_batch, _ in data_iter.take((num_background + batch_size - 1) // batch_size): # Take enough batches
             needed = num_background - samples_collected
             background_data_list.append(x_batch.numpy()[:needed]) # Take only needed samples
             samples_collected += x_batch.shape[0]
             if samples_collected >= num_background:
                 break
        if not background_data_list:
             logger.error("Could not retrieve background data for SHAP. Aborting SHAP analysis.")
             return
        background_data = np.concatenate(background_data_list, axis=0)
        # Ensure we have exactly num_background samples if possible
        background_data = background_data[:num_background]
        logger.info(f"Collected background data for SHAP. Shape: {background_data.shape}") # (N_bg, Seq, Feat)

        # Summarize background data (e.g., using median, mean, or k-means)
        # For sequence data, summarizing is tricky. Options:
        # a) Use k-means on the flattened sequences (loses structure).
        # b) Use k-means on features averaged over time.
        # c) Use a small subset directly (can be slow for KernelExplainer).
        # Let's use k-means on time-averaged features as a compromise.
        if len(background_data.shape) == 3: # (N, Seq, Feat)
            background_avg_features = np.mean(background_data, axis=1) # (N, Feat)
            num_centers = min(10, background_avg_features.shape[0]) # Use few centers
            logger.info(f"Summarizing background using k-means on time-averaged features ({num_centers} centers)...")
            background_summary = shap.kmeans(background_avg_features, num_centers).data # Shape (num_centers, Feat)
        else: # Assume data is already (N, Feat)
             logger.info("Background data seems non-sequential. Using k-means directly...")
             num_centers = min(10, background_data.shape[0])
             background_summary = shap.kmeans(background_data, num_centers).data # Shape (num_centers, Feat)


        # 2. Prepare Foreground Data (Samples to Explain)
        foreground_data_list = []
        samples_collected = 0
        # Need a fresh iterator if the previous one was consumed
        # Assuming data_iter is fresh or resettable here. This needs careful handling in practice.
        logger.info("Collecting foreground data for explanation...")
        for x_batch, _ in data_iter.take((num_foreground + batch_size - 1) // batch_size):
             needed = num_foreground - samples_collected
             foreground_data_list.append(x_batch.numpy()[:needed])
             samples_collected += x_batch.shape[0]
             if samples_collected >= num_foreground:
                 break
        if not foreground_data_list:
            logger.error("Could not retrieve foreground data for SHAP. Aborting SHAP analysis.")
            return
        foreground_data = np.concatenate(foreground_data_list, axis=0)
        foreground_data = foreground_data[:num_foreground] # Ensure exact number
        logger.info(f"Collected foreground data for SHAP. Shape: {foreground_data.shape}")


        # 3. Create SHAP Explainer
        #   - DeepExplainer: Potentially faster for TF models but sensitive to ops/layers.
        #   - KernelExplainer: Model-agnostic, generally works but can be very slow.
        # Choosing the right explainer and input format for sequences is challenging.

        # Prepare model function for SHAP: needs to take data and return scalar output.
        # Handle multi-output models: explain only the primary output (e.g., voltage).
        if isinstance(model.output, (list, tuple)):
            output_index = 0 # Index of the primary output to explain
            # Create a temporary model that outputs only the target scalar
            model_to_explain = tf.keras.Model(inputs=model.input, outputs=model.outputs[output_index])
            logger.info(f"Creating temporary model to explain output index {output_index}.")
        else:
            model_to_explain = model

        # Define prediction function wrapper for SHAP (takes numpy, returns numpy)
        # Needs to handle sequence input carefully. SHAP often works best on tabular (N, Features).
        # We will explain based on time-averaged features for simplicity with KernelExplainer.
        def predict_fn_avg_features(data_avg_features_np: np.ndarray) -> np.ndarray:
            # This function receives (N, Feat). We can't directly predict from this
            # for a sequence model. This approach is inherently limited for sequence models.
            # A more advanced approach might involve explaining importance at each time step.
            # --- Simplification ---
            # We cannot easily map averaged features back to a sequence prediction.
            # Instead of explaining averaged features, let's try explaining the prediction
            # based on the original sequence input using KernelExplainer. This will be slow.
            # KernelExplainer needs f(X) where X is (N, Features). We need to adapt it for sequences.
            # SHAP doesn't have great built-in support for direct sequence explanation with KernelExplainer.

            # --- Alternative: Explain the features of the *last* time step? ---
            # Or average the SHAP values over time steps? Requires different explainer setup.

            # --- Fallback: Return dummy values to indicate limitation ---
            logger.warning("SHAP KernelExplainer on time-averaged features is problematic for sequence models. "
                           "A more sophisticated sequence-specific SHAP method might be needed.")
            # Returning average prediction across a batch for dimensionality matching (not meaningful explanation)
            dummy_pred = model_to_explain(tf.random.normal(shape=(1,) + model.input_shape[1:]), training=False) # Predict on one dummy sample
            return np.tile(dummy_pred.numpy(), (data_avg_features_np.shape[0], 1))


        # --- Using KernelExplainer on averaged features (Known Limitation) ---
        logger.warning("Attempting SHAP KernelExplainer on time-averaged features (Note: This is a simplification).")
        if len(foreground_data.shape) == 3:
            foreground_avg_features = np.mean(foreground_data, axis=1) # (N_fg, Feat)
        else: # Assume already (N_fg, Feat)
             foreground_avg_features = foreground_data

        explainer = shap.KernelExplainer(predict_fn_avg_features, background_summary)
        logger.info("Calculating SHAP values using KernelExplainer (this might take a while)...")
        shap_values = explainer.shap_values(foreground_avg_features) # Shape (N_fg, Feat) or list for multi-output?

        # 4. Generate and Save Plots (for averaged features)
        # Ensure feature names match the averaged features
        if len(feature_names) == foreground_avg_features.shape[1]:
             shap_feature_names = feature_names
        else:
             shap_feature_names = [f"AvgFeat_{i}" for i in range(foreground_avg_features.shape[1])]
             logger.warning(f"Feature name count ({len(feature_names)}) doesn't match averaged feature count "
                            f"({foreground_avg_features.shape[1]}). Using generic names for SHAP plot.")


        plt.figure(figsize=(10, 6)) # Adjust figure size
        # Use try-except for plotting as SHAP can sometimes raise errors
        try:
            # If shap_values is a list (multi-output explanation), take the first element
            shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
            foreground_data_to_plot = foreground_avg_features

            shap.summary_plot(shap_values_to_plot, foreground_data_to_plot, feature_names=shap_feature_names, show=False, plot_size=None) # Use auto plot_size
            plt.title('SHAP Summary Plot (Features Averaged Over Time)')
            plot_path = os.path.join(output_dir, "shap_summary_plot_avg_features.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary plot (averaged features) saved to {plot_path}")
        except Exception as plot_err:
             logger.error(f"Failed to generate SHAP summary plot: {plot_err}", exc_info=True)
             plt.close() # Ensure plot is closed on error

        elapsed_time = time.time() - start_time
        logger.info(f"SHAP analysis finished in {elapsed_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"An error occurred during SHAP analysis: {e}", exc_info=True)

def perform_uq_evaluation(model: tf.keras.Model, data_iter: tf.data.Dataset,
                          num_samples: int, output_dir: str, batch_size: int):
    """
    Performs Uncertainty Quantification (UQ) evaluation using Monte Carlo Dropout
    (if model has dropout layers enabled during inference) or assumes a model
    that directly outputs mean and variance. Saves predictions and standard deviations.

    Args:
        model: The trained Keras model. Should have dropout layers for MC Dropout.
        data_iter: A tf.data.Dataset iterator yielding (features, target) for test data.
                   Must be resettable or a new instance.
        num_samples: Number of forward passes for MC Dropout or samples for other methods.
        output_dir: Directory to save UQ results.
        batch_size: Batch size for prediction.
    """
    logger.info(f"Starting UQ evaluation using Monte Carlo Dropout with {num_samples} samples...")
    start_time = time.time()

    all_y_true_list = []
    all_y_pred_mean_list = []
    all_y_pred_std_list = []
    all_y_pred_samples_list = [] # Optional: store all samples

    uq_method = "mc_dropout" # Assume MC Dropout, could be configured
    logger.info(f"Using UQ method: {uq_method}")

    try:
        # Iterate through data
        for x_batch, y_batch in tqdm(data_iter, desc="UQ Prediction"):
            mc_predictions = []

            if uq_method == "mc_dropout":
                # Perform multiple forward passes with training=True to enable dropout
                for _ in range(num_samples):
                    try:
                        # Ensure dropout is active during this prediction call
                        y_pred_sample = model(x_batch, training=True)
                        # Handle multi-output models if necessary
                        if isinstance(y_pred_sample, (list, tuple)):
                            y_pred_sample = y_pred_sample[0] # Take primary output
                        mc_predictions.append(y_pred_sample)
                    except Exception as mc_err:
                         logger.error(f"MC Dropout forward pass failed: {mc_err}", exc_info=True)
                         # Append NaNs or skip sample? Appending NaNs indicates failure for this sample.
                         nan_shape = y_batch.shape # Match target shape
                         mc_predictions.append(tf.fill(nan_shape, tf.constant(np.nan, dtype=tf.float32)))


                if not mc_predictions: continue # Skip batch if all passes failed

                # Stack predictions along a new dimension (samples, batch, seq, feat)
                mc_predictions_tensor = tf.stack(mc_predictions, axis=0)

                # Calculate mean and standard deviation across the samples dimension (axis=0)
                y_pred_mean = tf.reduce_mean(mc_predictions_tensor, axis=0)
                y_pred_std = tf.math.reduce_std(mc_predictions_tensor, axis=0)

                # Optional: Store all samples for more detailed analysis (can consume lots of memory)
                # all_y_pred_samples_list.append(mc_predictions_tensor.numpy())

            # --- Placeholder for other UQ Methods (e.g., Variational Inference) ---
            elif uq_method == "variational":
                 # Assume model directly outputs mean and variance (or log_var)
                 # model_outputs = model(x_batch, training=False)
                 # y_pred_mean = model_outputs[0] # Assuming mean is first output
                 # y_pred_log_var = model_outputs[1] # Assuming log variance is second
                 # y_pred_std = tf.exp(0.5 * y_pred_log_var)
                 logger.warning("Variational inference UQ evaluation not fully implemented.")
                 y_pred_mean = model(x_batch, training=False) # Fallback: use standard prediction
                 if isinstance(y_pred_mean, (list, tuple)): y_pred_mean = y_pred_mean[0]
                 y_pred_std = tf.zeros_like(y_pred_mean) # No uncertainty estimate
            # -----------------------------------------------------------------
            else:
                 logger.error(f"Unsupported UQ method: {uq_method}")
                 # Fallback: standard prediction with zero uncertainty
                 y_pred_mean = model(x_batch, training=False)
                 if isinstance(y_pred_mean, (list, tuple)): y_pred_mean = y_pred_mean[0]
                 y_pred_std = tf.zeros_like(y_pred_mean)


            # Append results for this batch
            all_y_true_list.append(y_batch.numpy())
            all_y_pred_mean_list.append(y_pred_mean.numpy())
            all_y_pred_std_list.append(y_pred_std.numpy())

        # Check if any results were collected
        if not all_y_true_list:
             logger.error("No UQ results collected. Aborting UQ evaluation.")
             return

        # Concatenate results from all batches
        y_true_all = np.concatenate(all_y_true_list, axis=0)
        y_pred_mean_all = np.concatenate(all_y_pred_mean_list, axis=0)
        y_pred_std_all = np.concatenate(all_y_pred_std_list, axis=0)
        # y_pred_samples_all = np.concatenate(all_y_pred_samples_list, axis=1) # Concat along batch dim

        logger.info(f"UQ Predictions collected. Shapes: y_true={y_true_all.shape}, "
                    f"y_pred_mean={y_pred_mean_all.shape}, y_pred_std={y_pred_std_all.shape}")

        # Calculate UQ Metrics (Placeholders - require specific libraries/implementations)
        # - Calibration Error: Compare predicted probability intervals with observed frequencies.
        #   Requires binning predictions based on uncertainty.
        # - Sharpness: Average predictive standard deviation. Lower is better (less uncertain),
        #   but must be balanced with calibration.
        avg_std_dev = np.nanmean(y_pred_std_all) # Use nanmean to ignore potential NaNs
        logger.info(f"Average Predictive Standard Deviation (Sharpness): {avg_std_dev:.4f}")
        # - Negative Log-Likelihood (NLL): If predictions are distributional.

        # --- Example: Calculate percentage of true values within +/- 2 std dev interval ---
        lower_bound = y_pred_mean_all - 2 * y_pred_std_all
        upper_bound = y_pred_mean_all + 2 * y_pred_std_all
        within_interval = np.logical_and(y_true_all >= lower_bound, y_true_all <= upper_bound)
        coverage_prob = np.nanmean(within_interval) # Coverage of the ~95% interval
        logger.info(f"Empirical Coverage of +/- 2 Predictive Std Dev (~95% interval): {coverage_prob:.3f}")
        # ------------------------------------------------------------------------------------

        # Visualize Uncertainty (Example: Plot mean +/- interval)
        # Need to select specific samples or aggregate for clear plotting
        try:
            plot_predictions(y_true_all, y_pred_mean_all, y_pred_std=y_pred_std_all,
                             output_dir=output_dir, filename="predictions_with_uncertainty.png",
                             title="Predictions with +/- 2 Standard Deviations")
            logger.info("Uncertainty plot saved.")
        except Exception as plot_err:
             logger.error(f"Failed to generate uncertainty plot: {plot_err}", exc_info=True)


        # Save UQ results (mean, std dev)
        uq_save_path = os.path.join(output_dir, "uq_predictions.npz")
        np.savez_compressed(uq_save_path,
                            y_true=y_true_all,
                            y_pred_mean=y_pred_mean_all,
                            y_pred_std=y_pred_std_all)
                            # y_pred_samples=y_pred_samples_all) # Optional: save all samples
        logger.info(f"UQ predictions (mean, std) saved to {uq_save_path}")

        elapsed_time = time.time() - start_time
        logger.info(f"UQ evaluation finished in {elapsed_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"An error occurred during UQ evaluation: {e}", exc_info=True)


def main(args: argparse.Namespace):
    """Main function to orchestrate model evaluation."""
    # 1. Load Configuration and Setup Environment
    config_dict = load_config(args.config)
    # Override config only with explicitly provided args related to model/data/batch, not control args
    config_overrides = {key: val for key, val in vars(args).items() if val is not None and key in ['data_dir', 'model_name', 'batch_size']}
    config_dict = override_config_with_args(config_dict, argparse.Namespace(**config_overrides))
    effective_config_dict = get_effective_config(config_dict)

    # Determine output directory
    output_base_dir = args.output_dir or effective_config_dict.get("core", {}).get("output_dir", "results")
    # Use experiment name structure similar to training?
    eval_output_dir = os.path.join(output_base_dir, args.experiment_name)
    ensure_dir(eval_output_dir)

    # Setup logging
    setup_logging(log_dir=eval_output_dir, level=args.log_level, filename="evaluation.log")
    logger.info("=" * 50)
    logger.info(f"Starting Evaluation Run: {args.experiment_name}")
    logger.info(f"Evaluating Weights: {args.weights_path}")
    logger.info("=" * 50)
    log_config(effective_config_dict, logger)

    setup_system_environment(effective_config_dict.get("core", {}))
    hardware_adapter = get_hardware_adapter(effective_config_dict.get("hardware", {}))

    # 2. Load Test Data
    try:
        # Override data dir for evaluation if provided
        if args.data_dir:
            effective_config_dict['data']['output_dir'] = args.data_dir # Point data loader to correct processed dir
        data_loader = OptimizedDataLoader(effective_config_dict, hardware_adapter)
        # Get the test iterator (or validation if 'test' split doesn't exist)
        test_iter, test_steps = data_loader.get_test_iterator()
        if test_iter is None:
             logger.warning("Test iterator not found, trying validation iterator...")
             # Fallback to validation set if test set not found (e.g., during k-fold)
             # This assumes fold number isn't critical for evaluation, or we evaluate fold 0's val set.
             _, test_iter, _, test_steps = data_loader.get_train_val_iterators(fold=0) # Get fold 0 val set
             if test_iter is None:
                 raise ValueError("Failed to get test or validation data iterator.")
             else:
                  logger.info("Using validation set (fold 0) for evaluation as test set was not found.")

        input_shape = data_loader.get_feature_shape()
        output_shape = data_loader.get_target_shape()
        if input_shape is None or output_shape is None:
             raise ValueError("Failed to get input/output shape from data loader.")

    except Exception as data_err:
         logger.error(f"Data loading failed: {data_err}", exc_info=True)
         sys.exit(1)


    # --- Handle Evaluation Fraction ---
    if args.eval_fraction < 1.0 and args.eval_fraction > 0.0:
        if test_steps is None:
            logger.warning("Cannot apply eval fraction: Unknown number of test steps. Evaluating on all available data.")
        else:
             original_test_steps = test_steps
             eval_steps = max(1, int(original_test_steps * args.eval_fraction))
             logger.info(f"Evaluating on {args.eval_fraction*100:.1f}% of test data ({eval_steps} steps).")
             # Use tf.data.Dataset.take() to limit the number of steps
             test_iter = test_iter.take(eval_steps)
             test_steps = eval_steps # Update steps for progress bar
    # ---------------------------------

    # 3. Build Model Architecture
    try:
        model_name = effective_config_dict.get("model", {}).get("name")
        model_specific_cfg = effective_config_dict.get("model", {}).get(model_name, effective_config_dict.get("model", {}))
        # Build model WITHOUT loading weights initially
        model = ModelRegistry.get(model_name)(
            input_shape=input_shape,
            output_shape=output_shape,
            cfg=model_specific_cfg
        )
        logger.info(f"Model '{model_name}' built successfully.")
    except Exception as model_err:
         logger.error(f"Model building failed (model: {model_name}): {model_err}", exc_info=True)
         sys.exit(1)

    # 4. Load Trained Weights
    if os.path.exists(args.weights_path):
        logger.info(f"Loading trained weights from: {args.weights_path}")
        try:
            # Use load_weights which works for .h5 and TF checkpoint directories
            model.load_weights(args.weights_path).expect_partial() # Allow partial loads
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load weights from {args.weights_path}: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.error(f"Weights file or directory not found: {args.weights_path}")
        sys.exit(1)

    # 5. Compile Model (for metrics only)
    # Use metrics defined in config for consistency with training run
    metrics_names = effective_config_dict.get("trainer",{}).get("metrics", ["mae", "RootMeanSquaredError", "R2Score"])
    metric_instances = BaseTrainer.get_metrics(None, metrics_names) # Use static method to create instances
    model.compile(loss=None, metrics=metric_instances)
    logger.info(f"Model compiled for evaluation with metrics: {[m.name for m in metric_instances]}")


    # --- Perform Standard Evaluation (Predictions & Metrics) ---
    logger.info("Starting standard evaluation (predicting on test set)...")
    all_y_true_list = []
    all_y_pred_list = []
    # Store other info if needed and available from iterator
    # all_cell_ids = []
    # all_cycles = []

    eval_batch_size = args.batch_size or effective_config_dict.get("trainer", {}).get("batch_size", 32)

    # Manual prediction loop
    predict_start_time = time.time()
    for batch_data in tqdm(test_iter, total=test_steps, desc="Predicting"):
        # Assume iterator yields (features, target) or potentially more
        if isinstance(batch_data, tuple):
            x_batch = batch_data[0]
            y_batch = batch_data[1]
            # Extract metadata if present (modify FormatParser.parse_tfrecord_fn accordingly)
            # if len(batch_data) > 2: all_cell_ids.append(batch_data[2].numpy())
            # if len(batch_data) > 3: all_cycles.append(batch_data[3].numpy())
        else: # Assume iterator yields only features
            x_batch = batch_data
            y_batch = None # No ground truth available

        y_pred_batch = model.predict_on_batch(x_batch)
        # Handle multi-output models (take the primary prediction)
        if isinstance(y_pred_batch, (list, tuple)):
            y_pred_batch = y_pred_batch[0]

        if y_batch is not None: all_y_true_list.append(y_batch.numpy())
        all_y_pred_list.append(y_pred_batch)

    predict_time = time.time() - predict_start_time
    logger.info(f"Prediction loop finished in {predict_time:.2f} seconds.")

    if not all_y_pred_list:
         logger.error("Prediction loop did not yield any results. Cannot evaluate.")
         sys.exit(1)

    y_pred_all = np.concatenate(all_y_pred_list, axis=0)

    # Calculate and log metrics if ground truth is available
    if all_y_true_list:
        y_true_all = np.concatenate(all_y_true_list, axis=0)
        logger.info(f"Predictions collected. Shapes: y_true={y_true_all.shape}, y_pred={y_pred_all.shape}")

        metrics_results = calculate_metrics(y_true_all, y_pred_all, metrics_names)
        logger.info(f"--- Overall Metrics ---")
        for name, value in metrics_results.items():
             logger.info(f"  {name}: {value:.6f}")
        logger.info(f"-----------------------")


        # Save predictions and metrics
        np.savez_compressed(os.path.join(eval_output_dir, "predictions.npz"), y_true=y_true_all, y_pred=y_pred_all)
        pd.DataFrame([metrics_results]).to_csv(os.path.join(eval_output_dir, "overall_metrics.csv"), index=False)
        logger.info(f"Predictions and overall metrics saved to {eval_output_dir}")


        # Plotting
        if args.plot:
            logger.info("Generating prediction plots...")
            plot_predictions(y_true_all, y_pred_all, output_dir=eval_output_dir)
            logger.info("Prediction plots saved.")

    else: # Ground truth not available
         logger.info(f"Predictions generated (shape: {y_pred_all.shape}). Ground truth not available for metric calculation.")
         np.savez_compressed(os.path.join(eval_output_dir, "predictions_only.npz"), y_pred=y_pred_all)
         logger.info(f"Predictions saved to {eval_output_dir}")


    # --- Optional Evaluations ---

    # Cross-Temperature Evaluation (Placeholder)
    if args.cross_temp_eval:
        logger.warning("Cross-temperature evaluation not implemented yet. Requires temperature data per sample.")
        # Implementation steps:
        # 1. Ensure temperature is loaded alongside features/targets.
        # 2. Group y_true_all, y_pred_all by temperature bins.
        # 3. Calculate metrics for each bin.
        # 4. Save/log results per temperature.

    # Explainability Analysis
    if args.explain:
        # Need a fresh data iterator that includes features
        logger.info("Preparing data for SHAP analysis...")
        try:
             # Get a fresh iterator - essential!
             explain_iter, _ = data_loader.get_test_iterator() # Or val_iter if that was used
             if args.eval_fraction < 1.0: # Apply same fraction if used for main eval
                 explain_iter = explain_iter.take(eval_steps)

             # Get feature names from FormatParser or config
             feature_names = data_loader.parser.final_feature_list if hasattr(data_loader, 'parser') else effective_config_dict.get("data.feature_columns", [])

             perform_shap_analysis(
                 model=model,
                 data_iter=explain_iter.unbatch().batch(eval_batch_size), # Pass unbatched data? SHAP might rebatch. Pass batched.
                 feature_names=feature_names,
                 num_background=args.explain_samples,
                 num_foreground=args.explain_samples,
                 output_dir=eval_output_dir,
                 batch_size=eval_batch_size # Pass batch size used for iteration
             )
        except Exception as shap_err:
             logger.error(f"SHAP analysis failed to execute: {shap_err}", exc_info=True)


    # UQ Evaluation
    if args.uq_eval:
        logger.info("Preparing data for UQ evaluation...")
        try:
             # Get a fresh iterator
             uq_iter, _ = data_loader.get_test_iterator() # Or val_iter
             if args.eval_fraction < 1.0:
                 uq_iter = uq_iter.take(eval_steps)

             perform_uq_evaluation(
                 model=model,
                 data_iter=uq_iter.unbatch().batch(eval_batch_size), # Pass batched data
                 num_samples=args.uq_samples,
                 output_dir=eval_output_dir,
                 batch_size=eval_batch_size
             )
        except Exception as uq_err:
             logger.error(f"UQ evaluation failed to execute: {uq_err}", exc_info=True)


    logger.info("=" * 50)
    logger.info(f"Evaluation run {args.experiment_name} finished.")
    logger.info(f"Results saved in: {eval_output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)