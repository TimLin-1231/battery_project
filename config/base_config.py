# battery_project/config/base_config.py

import os
from datetime import datetime

# --- Project Root ---
# Assume this file is in config/, so root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Basic Configuration ---
config = {
    'project_name': 'BatterySOH_RUL_Prediction_Pack',
    'experiment_name': f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    'seed': 42,

    # --- Battery Pack Configuration ---
    'pack_config': {
        'num_cells_series': 13,       # Number of cells in series
        'num_cells_parallel': 1,        # Assuming 1P configuration (update if needed)
        'nominal_capacity_ah': 6.2,   # Total pack nominal capacity in Ampere-hours
        'nominal_cell_voltage': 3.6, # Nominal voltage of a single cell (e.g., 3.6V or 3.7V for NMC/LFP)
        'nominal_voltage_pack': 13 * 3.6, # Estimated nominal pack voltage
        'connection': '13s1p',      # Configuration string
    },

    # --- Physics Configuration (for PINN Loss / Models) ---
    'physics_config': {
        # Cell/Material Properties (Potentially for Equivalent Single Cell Model - ESCM)
        'initial_soc': 1.0,         # Initial State of Charge
        'cs_max_nominal': 51410,    # Max concentration in solid (mol/m^3) - Example for NMC, adjust for your cell type
        'faraday_const': 96485.3321,# Faraday constant (C/mol)
        'r_gas_const': 8.31446,     # Ideal gas constant (J/(mol*K))
        # Add other relevant physical parameters (diffusion coeffs, reaction rates, R_ohm_cell, etc.)
        # These might be fixed, loaded, or made trainable.
        'nominal_pack_resistance_ohm': 0.050, # Estimated initial pack internal resistance (50 mOhm example) - Update based on data
        'ocv_soc_lookup_path': os.path.join(PROJECT_ROOT, 'data', 'lookups', 'ocv_soc_curve.csv'), # Path to OCV-SOC lookup table (for ESCM)
    },

    # --- Data Configuration ---
    'data': {
        'raw_path': os.path.join(PROJECT_ROOT, 'data', 'raw'),
        'interim_path': os.path.join(PROJECT_ROOT, 'data', 'interim'), # For cleaned data
        'processed_path': os.path.join(PROJECT_ROOT, 'data', 'processed'), # For features, sequences, tfrecords
        'results_path': os.path.join(PROJECT_ROOT, 'results'),
        'log_path': os.path.join(PROJECT_ROOT, 'logs'),
        'model_save_path': os.path.join(PROJECT_ROOT, 'models', 'saved'),

        # --- Feature & Target Columns ---
        # !!! IMPORTANT: Update these based on your actual CSV column names for PACK data !!!
        'features': [
            'pack_voltage',         # Example: Pack terminal voltage
            'pack_current',         # Example: Pack current (positive=charge, negative=discharge)
            'pack_temperature_avg', # Example: Average or representative pack temperature
            'cycle_time_s',         # Example: Time within the cycle
            # Add other relevant pack-level features if available
            # 'ambient_temperature',
            # --- Features from Feature Engineering (Pack Level) ---
            # 'pack_resistance_estimate',
            # 'coulombic_efficiency',
            # 'voltage_curve_skewness', # etc.
        ],
        'target': ['pack_soh'],      # Example: Target variable (State of Health of the pack)
        'time_column': 'timestamp', # Original timestamp column name
        'cycle_column': 'cycle_index',# Cycle number column name
        'step_column': 'step_type',   # Column indicating charge/discharge/rest

        # --- Preprocessing ---
        'voltage_range': [2.5 * 13, 4.2 * 13], # Approximate pack voltage range (13S)
        'current_range': [-10.0, 10.0], # Example pack current range (Amps) - Adjust based on C-rate
        'temperature_range': [0, 60],   # Example pack temperature range (Celsius)
        'outlier_std_threshold': 3.0,
        'charge_discharge_threshold': 0.1, # Current threshold to define charge/discharge steps (Amps)

        # --- TFRecord/DataLoader Settings ---
        'tfrecord_compression': 'GZIP',
        'shuffle_buffer_size': 1000,

        # --- Data Augmentation ---
        'augmentation': {
            'enabled': True, # Global switch for augmentation during training
            'noise_level': 0.005, # Stddev of Gaussian noise added to scaled features
            'time_warp_scale': 0.0, # Scale for time warping (0 = disabled)

            'gan': {
                'enabled': True, # Enable GAN-based augmentation
                'probability': 0.3, # Probability of applying GAN augmentation to a sample
                'generator_path': os.path.join(PROJECT_ROOT, 'models', 'saved', 'rcgan_generator.keras'), # Path to trained RCGAN generator
                 # --- GAN Specifics (Ensure matches trained generator) ---
                 'conditional_dim': 1, # Dimension of the condition vector (e.g., target SOH)
                 'noise_dim': 100,     # Dimension of the noise vector input
                 'output_features': [ # List of features the GAN *generates* (must match order/count)
                     'pack_voltage', 'pack_current', 'pack_temperature_avg', 'cycle_time_s' # Example
                 ]
            },
             # Add other augmentation techniques if needed (e.g., cutout, mixup)
        }
    },

    # --- Model Configuration ---
    'model': {
        'name': 'cnn_bilstm_attention', # Options: 'cnn_bilstm_attention', 'pinn', 'gan', 'surrogate'
        'sequence_length': 100,       # Input sequence length (number of time steps)
        'prediction_length': 1,        # Output sequence length (usually 1 for SOH/RUL)

        # --- CNN-BiLSTM-Attention Specifics ---
        'cnn_filters': [32, 64, 64],
        'cnn_kernel_size': 3,
        'bilstm_units': 64,
        'attention_units': 32,
        'dropout_rate': 0.2,
        'final_dense_units': [32],     # Units in dense layers before final output

        # --- PINN Specifics ---
        'pinn': {
            'architecture': 'lstm_rc', # Options: 'lstm_rc', 'mlp_ode', etc.
            'lstm_units': 50,
            'rc_pairs': 2,            # Number of RC pairs in equivalent circuit layers
            'state_dims': {           # Dimensions/names of predicted internal states (besides voltage)
                'soc_avg_equiv': 1,   # Example: Equivalent average SOC
                'eta_pack': 1,        # Example: Equivalent pack overpotential
                # Add other states your PINN model predicts
            },
            'output_structure': 'dict', # 'dict' or 'tuple' - How model returns multiple outputs
        },

         # --- GAN Specifics (Discriminator/Generator HParams) ---
         'gan': {
             'generator': {
                 'lstm_units': 128,
                 'noise_dim': 100, # Should match data->augmentation->gan->noise_dim
                 'conditional_dim': 1, # Should match data->augmentation->gan->conditional_dim
                 'output_dim': 4, # Number of features GAN should generate
             },
             'discriminator': {
                 'cnn_filters': [64, 128],
                 'cnn_kernel_size': 5,
                 'lstm_units': 64,
                 'dropout_rate': 0.3,
             }
         },

         # --- Surrogate Model Specifics ---
         'surrogate': {
             # Define surrogate model architecture (e.g., similar to baseline)
             'cnn_filters': [32, 64],
             'cnn_kernel_size': 3,
             'lstm_units': 64,
             # ...
         }
    },

    # --- Training Configuration ---
    'training': {
        'optimizer': 'adam',        # Optimizer name (tf.keras.optimizers)
        'learning_rate': 1e-3,
        'batch_size': 64,           # Adjust based on GPU memory
        'epochs': 100,
        'loss_function': 'mean_squared_error', # Default loss for baseline/surrogate
                                              # Will be overridden by specific trainers (PINN, GAN)
        'metrics': ['mean_absolute_error', 'root_mean_squared_error'], # Additional metrics
        'validation_split': 0.15,   # Fraction of training data for validation
        'test_split': 0.15,         # Fraction of data for testing (applied after train/val split)
        'split_strategy': 'group_k_fold', # 'random', 'group_k_fold' (by battery ID/group), 'leave_one_out'
        'num_folds': 5,             # For k-fold strategies
        'group_column': 'battery_id',# Column identifying battery groups for splitting

        # --- Callbacks ---
        'early_stopping': {
            'enabled': True,
            'monitor': 'val_loss',
            'patience': 15,
            'min_delta': 1e-4,
            'restore_best_weights': True,
        },
        'model_checkpoint': {
            'enabled': True,
            'monitor': 'val_loss',
            'save_best_only': True,
            'save_weights_only': False, # Save full model
        },
        'reduce_lr_on_plateau': {
            'enabled': True,
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6,
        },
        'tensorboard': {
            'enabled': True,
            'log_dir': None, # Automatically set based on experiment name
            'histogram_freq': 1, # Log histograms of weights/biases
            'profile_batch': '50,100', # Profile batches 50-100 for performance analysis
        },

        # --- Transfer Learning ---
        'transfer_learning': {
            'enabled': False,
            'source_model_path': None, # Path to pre-trained source model
            'target_datasets': [],    # List of target dataset identifiers
            'fine_tune_layers': -1,   # Number of layers to unfreeze (-1 = all)
            'fine_tune_epochs': 20,
            'fine_tune_lr': 1e-4,
            'fine_tune_batch_size': 32,
            'freeze_batch_norm': True,# Keep Batch Norm layers frozen during fine-tuning
        },

         # --- PINN Specific Training ---
        'pinn_training': {
            'loss_config': { # Configuration for EnhancedPhysicsLoss weights
                'lambda_data': 1.0,
                'lambda_pde': 0.1,
                'lambda_bc_ic': 0.1,
                'lambda_bv': 0.05,
                'lambda_cons': 0.05,
                'lambda_tau': 0.01,
                'lambda_consist': 0.01,
                'lambda_smooth': 0.005,
                # Add physics params needed by loss here if not in global physics_config
                # 'initial_soc': 1.0, # Example, redundant if in physics_config
            },
            'non_dimensionalization': True, # Enable non-dimensionalization in PinnTrainer
            'scale_factors': { # Provide typical scales OR let trainer calculate them
                 # 'time': 3600.0, # Example: scale time by seconds in an hour
                 # 'voltage': 3.7 * 13, # Example: scale voltage by nominal pack voltage
                 # 'current': 6.2, # Example: scale current by nominal capacity (1C)
                 # 'temperature': 50.0 # Example: scale temperature by max expected range
                 # Add scales for internal states if predicted/used
                 # 'soc': 1.0,
            }
        },

         # --- GAN Specific Training ---
        'gan_training': {
            'generator_lr': 1e-4,
            'discriminator_lr': 1e-4,
            'gp_weight': 10.0,       # Weight for gradient penalty (WGAN-GP)
            'n_critic': 5,           # Train discriminator n times per generator train step
            'label_smoothing': 0.0,  # Label smoothing for discriminator (0 = disabled)
        }
    },

    # --- Evaluation Configuration ---
    'evaluation': {
        'batch_size': 128, # Can often use larger batch size for evaluation
        'metrics': ['mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error', 'r2_score'],
        'cross_condition': { # Evaluate robustness across different conditions
            'enabled': True,
            'condition_column': 'temperature_condition', # Column defining the condition (e.g., temp range, C-rate)
            'conditions_to_test': ['low_temp', 'high_temp', 'high_c_rate'] # Specific conditions to report separately
        },
        'small_sample': { # Evaluate performance with limited target data during fine-tuning/training
             'enabled': True,
             'fractions': [0.1, 0.2], # Fractions of target data to use (e.g., 10%, 20%)
        },
        'interpretability': {
            'enabled': True,
            'method': 'shap', # 'shap' or 'lime'
            'num_explain_samples': 50, # Number of test samples to explain
            'shap_explainer': 'Gradient', # Options for SHAP: 'Gradient', 'Deep', 'Kernel' (Kernel is slow)
        },
        'uncertainty': {
             'enabled': True,
             'method': 'mc_dropout', # 'mc_dropout', 'ensemble'
             'mc_samples': 50, # Number of Monte Carlo samples for MC Dropout
        },
        'physics_eval': { # For PINN models
            'enabled': True,
            'check_residuals': True, # Calculate and report average physics residuals on test set
            'check_physical_bounds': True, # Check if predicted states stay within reasonable physical limits
        }
    },

    # --- Hyperparameter Optimization (HPO) ---
    'hpo': {
        'enabled': False,
        'search_algorithm': 'hyperopt', # 'hyperopt', 'optuna', 'bayesian'
        'scheduler': 'asha',          # 'asha', 'fifo', 'pbt'
        'num_samples': 50,            # Number of hyperparameter configurations to try
        'metric_to_optimize': 'val_loss',
        'mode': 'min',               # 'min' or 'max'
        'cpu_per_trial': 2,
        'gpu_per_trial': 0.5,        # Request fractional GPU if needed
        'max_concurrent_trials': 4,
        'grace_period': 5,           # ASHA scheduler parameter (min epochs before stopping)
        'reduction_factor': 2,       # ASHA scheduler parameter
        'hpo_config_path': os.path.join(PROJECT_ROOT, 'config', 'hpo_search_space.py')
    },

    # --- System / Hardware Configuration ---
    'system': {
        'mixed_precision': True,      # Use mixed precision (float16) for potential speedup
        'jit_compile': False,         # Use XLA compilation (tf.function(jit_compile=True)) - can be faster but less flexible
        'gpu_memory_growth': True,    # Set memory growth for GPUs to avoid allocating all memory at once
        'deterministic_ops': False,   # Use deterministic ops for reproducibility (can be slower)
        # Hardware specific settings are usually detected/set by system_config.py
        'hardware_config_path': os.path.join(PROJECT_ROOT, 'config', 'system_config.py'),
    }
}

# --- Function to Access Config ---
def get_config():
    # Potentially load overrides from a YAML file or environment variables here
    return config

# --- Example Usage ---
if __name__ == '__main__':
    cfg = get_config()
    print("Project Name:", cfg['project_name'])
    print("Pack Config:", cfg['pack_config'])
    print("Data Features:", cfg['data']['features'])
    print("Target:", cfg['data']['target'])
    print("PINN Loss Config:", cfg['training']['pinn_training']['loss_config'])
    print("Model Save Path:", cfg['data']['model_save_path'])