# battery_project/trainers/pinn_trainer.py

import tensorflow as tf
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple, Any

from .base_trainer import BaseTrainer
from .losses import EnhancedPhysicsLoss  # Import the revised loss
from ..models.pinn import build_pinn_pack_model # Import the revised model builder

logger = logging.getLogger(__name__)

class PinnTrainer(BaseTrainer):
    """
    Trainer specifically designed for Physics-Informed Neural Networks (PINNs)
    for battery pack modeling. Overrides train_step and test_step to handle
    physics loss calculations and model input/output structures.

    Assumes data loader provides normalized data.
    Relies on EnhancedPhysicsLoss to handle internal rescaling using scale factors
    provided in the configuration.
    """

    def __init__(self, config: Dict, strategy: Optional[tf.distribute.Strategy] = None):
        """
        Initializes the PinnTrainer.

        Args:
            config (Dict): The main configuration dictionary.
            strategy (Optional[tf.distribute.Strategy]): Distribution strategy.
        """
        self.pinn_config = config.get('training', {}).get('pinn_training', {})
        self.scale_factors = self.pinn_config.get('scale_factors', {})
        self.do_nondim = self.pinn_config.get('non_dimensionalization', True) # Flag from config

        if self.do_nondim and not self.scale_factors:
            logger.warning("Non-dimensionalization enabled, but scale_factors are missing in config. Physics loss might be incorrect.")
            # Ideally, calculate scales from data here or ensure they are provided.
            # For now, loss function will default to scales of 1.0 if missing.

        # Inject scale_factors into the main config if not already present at the expected location for the loss
        # Ensure the loss function receives the scales correctly.
        if 'training' not in config: config['training'] = {}
        if 'pinn_training' not in config['training']: config['training']['pinn_training'] = {}
        config['training']['pinn_training']['scale_factors'] = self.scale_factors
        logger.info(f"PinnTrainer initialized. Using Scale Factors: {self.scale_factors}")

        # --- Find time feature index ---
        self.feature_list = config.get('data', {}).get('features', [])
        try:
            # Assuming time input feature name is 'time_input' or similar standard name used in config
            # Need a consistent way to identify the time feature column
            # Let's assume it's the last feature if not specified? Risky.
            # Best: Add a specific key in config like 'time_feature_name': 'time_input_norm'
            time_feature_name = config.get('data', {}).get('time_feature_name', 'time_input_norm') # Example standard name
            # Find index based on the *actual* feature list used
            self.time_feature_index = self.feature_list.index(time_feature_name)
            logger.info(f"Identified time feature '{time_feature_name}' at index {self.time_feature_index}")
        except ValueError:
            logger.warning(f"Time feature '{time_feature_name}' not found in feature list: {self.feature_list}. "
                           f"Physics loss gradient calculations might fail. Assuming index -1 (last feature).")
            self.time_feature_index = -1 # Fallback: Assume time is the last feature

        # Identify indices for other inputs needed by loss's model_inputs dict
        self.input_indices = {}
        required_inputs = ['current_input', 'temperature_input', 'time_input'] # Keys expected by loss's _get_required_states
        input_feature_map = { # Map required input keys to actual feature names in self.feature_list
            'current_input': config.get('data',{}).get('current_feature_name', 'pack_current_norm'), # Example name
            'temperature_input': config.get('data',{}).get('temperature_feature_name', 'pack_temperature_avg_norm'), # Example name
            'time_input': time_feature_name
        }
        for key, feature_name in input_feature_map.items():
            try:
                self.input_indices[key] = self.feature_list.index(feature_name)
            except ValueError:
                 logger.error(f"Required input feature '{feature_name}' (for loss key '{key}') not found in feature list: {self.feature_list}")
                 # Decide how to handle: raise error, or skip physics term?
                 raise ValueError(f"Missing required input feature for physics loss: {feature_name}")


        super().__init__(config, strategy) # Call parent init AFTER setting up scales/indices


    def _build_model(self) -> tf.keras.Model:
        """Builds the PINN model using the specific pack model builder."""
        logger.info("Building PINN Pack model...")
        # Ensure the correct builder is called
        return build_pinn_pack_model(self.config)


    def _get_loss(self) -> tf.keras.losses.Loss:
        """Gets the physics-informed loss function."""
        logger.info("Initializing EnhancedPhysicsLoss for Pack ESCM.")
        # Pass the full config, which includes scales injected in __init__
        return EnhancedPhysicsLoss(self.config)


    # --- Override train_step and test_step ---
    @tf.function
    def train_step(self, data):
        """
        Performs a single training step with physics-informed loss.
        Overrides the default train_step from BaseTrainer.
        """
        x, y_true_norm = data # Assumes data loader provides normalized x and y_true

        with tf.GradientTape() as tape:
            # --- Watch Time Input (for d(state)/dt gradient calculation) ---
            # Need to explicitly watch the time column within the input tensor 'x'
            # tape.watch(x[:, :, self.time_feature_index]) # Watch the time slice? Less direct.
            # Better approach: Ensure model uses time input, tape watches model's trainable vars

            # --- Forward pass ---
            # Model takes normalized inputs
            y_pred_struct_norm = self.model(x, training=True)

            # --- Prepare inputs for the loss function ---
            # Loss function expects a dictionary of normalized inputs
            model_inputs_norm = {
                key: x[:, :, idx:idx+1] # Extract corresponding slice, keep dims (batch, time, 1)
                for key, idx in self.input_indices.items()
            }
            # Verify shapes if needed: tf.print("Time input shape for loss:", tf.shape(model_inputs_norm['time_input']))

            # --- Calculate Loss ---
            # Loss function takes normalized y_true, normalized y_pred dict, and normalized model_inputs dict
            # It performs rescaling internally using scale_factors from config.
            # TODO (Advanced): Calculate d(soc)/dt gradient here using tape.gradient
            # dsoc_dt_norm = tape.gradient(y_pred_struct_norm['soc_avg_equiv'], model_inputs_norm['time_input'])
            # Handle None gradient, then pass dsoc_dt_norm to loss function. Requires modifying loss signature.
            loss_value = self.loss(y_true_norm, y_pred_struct_norm, model_inputs=model_inputs_norm)

            # Apply regularization losses (if any defined directly in the model)
            if self.model.losses:
                 loss_value += tf.add_n(self.model.losses)

            # Handle mixed precision scaling
            scaled_loss = self.optimizer.get_scaled_loss(loss_value) if self.use_mixed_precision else loss_value

        # --- Compute Gradients ---
        # Use scaled loss for gradient calculation with mixed precision
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        # Unscale gradients if using mixed precision
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients) if self.use_mixed_precision else scaled_gradients

        # --- Apply Gradients ---
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # --- Update Metrics ---
        # Update metrics using the primary target (voltage) and its prediction
        # Metrics operate on the normalized scale here
        self.compiled_metrics.update_state(y_true_norm, y_pred_struct_norm['voltage']) # Assuming y_true_norm corresponds to voltage

        # Return metric results dictionary
        return {m.name: m.result() for m in self.metrics}


    @tf.function
    def test_step(self, data):
        """
        Performs a single evaluation step.
        Overrides the default test_step from BaseTrainer.
        """
        x, y_true_norm = data

        # --- Forward pass ---
        y_pred_struct_norm = self.model(x, training=False)

        # --- Prepare inputs for the loss function ---
        model_inputs_norm = {
            key: x[:, :, idx:idx+1]
            for key, idx in self.input_indices.items()
        }

        # --- Calculate Loss ---
        # No gradient calculation needed here for loss value itself
        loss_value = self.loss(y_true_norm, y_pred_struct_norm, model_inputs=model_inputs_norm)

        # Add regularization losses if any
        if self.model.losses:
             loss_value += tf.add_n(self.model.losses)

        # --- Update Metrics ---
        self.compiled_metrics.update_state(y_true_norm, y_pred_struct_norm['voltage'])

        # Return metric results dictionary
        return {m.name: m.result() for m in self.metrics}


    # --- Non-dimensionalization Helpers (Less critical if loss handles rescaling) ---
    # Keep these commented out unless needed for explicit pre/post scaling outside the loss function

    # def _get_scale_factors(self) -> Dict[str, float]:
    #     """Determines scale factors for non-dimensionalization."""
    #     if self.scale_factors:
    #         logger.info("Using pre-defined scale factors from config.")
    #         return self.scale_factors
    #     else:
    #         # TODO: Implement calculation from training data statistics if needed
    #         logger.warning("Scale factors not found in config and calculation from data not implemented. Using defaults (1.0).")
    #         # Populate with defaults based on expected features/states
    #         default_scales = {'time': 1.0, 'voltage': 1.0, 'current': 1.0, 'temperature': 1.0, 'soc_equiv': 1.0, 'resistance_pack': 1.0}
    #         return default_scales

    # def _rescale_input(self, x_norm: tf.Tensor) -> tf.Tensor:
    #     """Rescales normalized model input 'x' back to physical units."""
    #     # ... Implementation would use self.scale_factors based on feature order ...
    #     pass

    # def _rescale_output(self, y_pred_norm_struct: Dict) -> Dict:
    #     """Rescales normalized model output dictionary back to physical units."""
    #     # ... Implementation would use self.scale_factors for each predicted state ...
    #     pass


# --- Example Usage ---
if __name__ == '__main__':
    print("Testing PinnTrainer setup...")
    # Create dummy config similar to the one used for model testing
    dummy_config = {
        'data': {
            'features': ['pack_voltage_norm', 'pack_current_norm', 'pack_temperature_avg_norm', 'time_input_norm'],
            'target': ['pack_voltage_norm'],
            'time_feature_name': 'time_input_norm', # Specify time feature name
            'current_feature_name': 'pack_current_norm',
            'temperature_feature_name': 'pack_temperature_avg_norm',
        },
        'model': {
            'name': 'pinn', # Important: Select PINN model type
            'sequence_length': 50,
            'prediction_length': 1,
            'pinn': {
                'lstm_units': 40,
                'output_structure': 'dict',
                # state_dims not directly used by builder now
            }
        },
        'training': {
            'optimizer': 'adam',
            'learning_rate': 1e-4,
            'batch_size': 4,
             'metrics': ['mean_absolute_error'], # Example metric
            'pinn_training': {
                'non_dimensionalization': True,
                'scale_factors': { # Provide dummy scales
                     'time': 3600.0,
                     'voltage': 50.0, # Approx pack voltage range / nominal
                     'current': 10.0, # Approx current range / nominal
                     'temperature': 50.0,
                     'soc_equiv': 1.0,
                     'resistance_pack': 0.1 # Approx pack resistance scale (Ohm)
                },
                'loss_config': { # Weights for the loss terms
                    'lambda_data': 1.0,
                    'lambda_soc_ode': 0.2,
                    'lambda_voltage_eqn': 0.2,
                    'lambda_bc_ic': 0.1,
                    'lambda_resistance_phys': 0.05,
                    'lambda_cons': 0.1,
                    'lambda_smooth': 0.01,
                }
            }
        },
         'physics_config': { # Need path for OCV curve
              'ocv_soc_lookup_path': './dummy_ocv.csv' # Create dummy OCV file
         },
         'pack_config': {'num_cells_series': 13, 'nominal_capacity_ah': 6.2},
         'system': {'mixed_precision': False}
    }

    # Create dummy OCV file
    ocv_data = {'soc': np.linspace(0, 1, 11), 'ocv': np.linspace(3.0, 4.2, 11)} # Simple linear OCV
    pd.DataFrame(ocv_data).to_csv(dummy_config['physics_config']['ocv_soc_lookup_path'], index=False)
    print(f"Created dummy OCV file: {dummy_config['physics_config']['ocv_soc_lookup_path']}")


    # --- Instantiate Trainer ---
    try:
        # No distribution strategy for basic test
        trainer = PinnTrainer(config=dummy_config, strategy=None)

        # --- Test Build Process ---
        assert trainer.model is not None
        assert isinstance(trainer.loss, EnhancedPhysicsLoss)
        assert trainer.optimizer is not None
        # Check if scale factors are passed to loss
        assert trainer.loss.scale_factors == dummy_config['training']['pinn_training']['scale_factors']

        print("\nTrainer Initialization successful.")

        # --- Test train_step / test_step execution (requires dummy data) ---
        print("\nTesting train_step execution...")
        batch_size = dummy_config['training']['batch_size']
        seq_len = dummy_config['model']['sequence_length']
        num_features = len(dummy_config['data']['features'])
        pred_len = dummy_config['model']['prediction_length'] # Should be 1 for voltage target

        # Create one batch of dummy normalized data
        x_norm = tf.random.uniform((batch_size, seq_len, num_features), dtype=tf.float32)
        # Ensure y_true_norm matches the primary target (voltage) shape
        y_true_norm = tf.random.uniform((batch_size, seq_len, pred_len), dtype=tf.float32) # Target matches voltage output shape


        # Compile the model (needed for metrics tracking)
        trainer.compile()

        # Execute train_step
        metrics_result = trainer.train_step((x_norm, y_true_norm))
        print("train_step metrics:", metrics_result)
        assert 'loss' in metrics_result # Base loss tracked by compile
        assert dummy_config['training']['metrics'][0] in metrics_result # Custom metric

        # Execute test_step
        print("\nTesting test_step execution...")
        metrics_result_test = trainer.test_step((x_norm, y_true_norm))
        print("test_step metrics:", metrics_result_test)
        assert 'loss' in metrics_result_test
        assert dummy_config['training']['metrics'][0] in metrics_result_test

        print("\nPinnTrainer train/test step execution test completed.")

    except Exception as e:
         print(f"\nError during PinnTrainer test: {e}", exc_info=True)
    finally:
         # Clean up dummy OCV file
         if os.path.exists(dummy_config['physics_config']['ocv_soc_lookup_path']):
             os.remove(dummy_config['physics_config']['ocv_soc_lookup_path'])