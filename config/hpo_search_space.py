# battery_project/config/hpo_search_space.py
"""
Configuration file for Hyperparameter Optimization (HPO) search space using Ray Tune.
This file defines the hyperparameters to tune and their search ranges/distributions.
It is loaded by scripts/train.py when the --hpo flag is used.
"""

import logging

# --- Optional Ray Tune Import ---
# Attempt to import Ray Tune; define placeholders if not available.
try:
    from ray import tune
    import numpy as np # Often useful for defining ranges
    RAY_AVAILABLE = True
    logging.info("Ray Tune library found for HPO search space definition.")
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray Tune library not found. Using placeholder values for HPO search space. "
                    "Install with: pip install 'ray[tune]'")
    # Define placeholders for tune primitives if Ray Tune is not installed
    class TunePlaceholder:
        def uniform(self, lower, upper): return (lower + upper) / 2.0
        def loguniform(self, lower, upper): return np.exp((np.log(lower) + np.log(upper)) / 2.0)
        def choice(self, categories): return categories[0] if categories else None
        def grid_search(self, values): return values[0] if values else None
        def quniform(self, lower, upper, q): return round(((lower + upper) / 2.0) / q) * q
        def qloguniform(self, lower, upper, q): return round((np.exp((np.log(lower) + np.log(upper)) / 2.0)) / q) * q
    tune = TunePlaceholder()
# ---------------------------


# === Define the Hyperparameter Search Space ===
# Keys in this dictionary should correspond to parameters you want Ray Tune to vary.
# The values use `tune` functions (e.g., tune.uniform, tune.loguniform, tune.choice).
# Structure the keys carefully, potentially matching the structure in base_config.py
# for easier merging in the train_function_for_tune (scripts/train.py).
# Using dot notation in keys (e.g., "trainer.lr") might require specific merging logic.
# A flatter structure might be simpler to merge.

search_space = {

    # --- Trainer Hyperparameters ---
    "trainer.lr": tune.loguniform(1e-5, 5e-3), # Learning rate (log scale recommended)
    "trainer.batch_size": tune.choice([16, 32, 64, 128]), # Batch size options
    # "trainer.optimizer": tune.choice(["adam", "rmsprop"]), # Example: Tune optimizer type
    # "trainer.patience": tune.choice([10, 15, 20]), # Example: Tune early stopping patience

    # --- Model Agnostic Hyperparameters (if applicable) ---
    # These might be located directly under 'model' or within a specific model's section
    "model.dropout": tune.uniform(0.05, 0.4), # General dropout rate
    # "model.l2_reg": tune.loguniform(1e-6, 1e-3), # General L2 regularization

    # --- CNN-BiLSTM-Attention Specific Hyperparameters ---
    # Prefix keys with model name if base_config is structured that way,
    # or use flat keys and handle merging in train script. Using flat keys here for simplicity example.
    # Adjust these keys based on your final config structure and train_function_for_tune merging logic.
    "model.cnn_bilstm_attention.cnn_dropout": tune.uniform(0.05, 0.3),
    # "model.cnn_bilstm_attention.rnn_dropout": tune.uniform(0.1, 0.4), # If separate RNN dropout exists
    "model.cnn_bilstm_attention.attention_dropout": tune.uniform(0.05, 0.3),
    # Tuning number of units requires careful handling in model building:
    # "model.cnn_bilstm_attention.rnn_units_list": tune.choice([[64], [64, 64], [128], [128, 64]]), # Tune RNN structure
    # "model.cnn_bilstm_attention.final_dense_units": tune.choice([[32], [64], [32, 16]]), # Tune output head structure
    # Tuning kernel sizes:
    # "model.cnn_bilstm_attention.cnn_kernels": tune.choice([[5, 3], [7, 5, 3], [9, 7, 5]]),

    # --- GAN Specific Hyperparameters ---
    # "model.gan.generator.rnn_units_list": tune.choice([[128], [128, 64]]),
    # "model.gan.discriminator.cnn_filters": tune.choice([[32, 64], [64, 128, 256]]),
    # "trainer.gan.generator_lr": tune.loguniform(5e-5, 5e-4),
    # "trainer.gan.discriminator_lr": tune.loguniform(5e-5, 5e-4),
    # "trainer.gan.gp_weight": tune.uniform(5.0, 15.0),
    # "trainer.gan.discriminator_steps": tune.choice([3, 5, 7]),

    # --- PINN Specific Hyperparameters ---
    # "model.pinn.rnn_units": tune.choice([[32, 32], [64, 64]]),
    # "model.pinn.rc_dense_units": tune.choice([[16], [32], [32, 16]]),
    # Tuning loss weights (can be sensitive)
    # "trainer.pinn.pde_weight": tune.loguniform(1e-4, 1.0),
    # "trainer.pinn.bc_weight": tune.loguniform(1e-1, 10.0),
    # "trainer.pinn.tau_weight": tune.loguniform(1e-3, 1.0),

}

# === End of Search Space Definition ===

# Log a confirmation message if Ray Tune is available
if RAY_AVAILABLE:
    logging.info(f"HPO search space defined with {len(search_space)} parameters.")
    # You can add validation here to check if keys align with your config structure if needed
else:
    # This part executes only if Ray Tune couldn't be imported
    print("-" * 60)
    print("WARNING: Ray Tune ('ray[tune]') not installed.")
    print("The HPO search space defined in this file uses placeholder values.")
    print("Please install Ray Tune to enable hyperparameter optimization.")
    print("-" * 60)

# You can define multiple search spaces here and choose one in the train script if needed.
# Example:
# search_space_small = { ... }
# search_space_large = { ... }