# battery_project/trainers/losses.py

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import logging
import os
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# --- Physical Constants (Example Values) ---
# These should be reviewed and potentially loaded from config more robustly
FARADAY_CONST = 96485.3321 # C/mol
R_GAS_CONST = 8.31446 # J/(mol*K)


class EnhancedPhysicsLoss(tf.keras.losses.Loss):
    """
    Custom loss function for Physics-Informed Neural Networks (PINNs)
    applied to battery PACK modeling. Uses an Equivalent Single Cell Model (ESCM)
    approach combined with pack-level equations.

    Assumes:
    - The PINN model predicts pack-level quantities (V_pack_pred) and
      necessary equivalent internal states (e.g., soc_avg_equiv_pred, R_pack_effective_pred).
    - Model inputs include pack-level measurements and time 't'.
    - Configuration provides pack parameters (N_series, Q_pack) and OCV curve path.
    - Trainer provides scale factors for non-dimensionalization.
    """
    def __init__(self, config: Dict, model: tf.keras.Model = None, name="enhanced_physics_loss"):
        super().__init__(name=name)
        self.config = config # Full config for access to pack/physics/training sections
        self.loss_config = config.get('training', {}).get('pinn_training', {}).get('loss_config', {})
        self.physics_config = config.get('physics_config', {})
        self.pack_config = config.get('pack_config', {})
        self.model = model # Avoid using model directly in loss if possible

        # --- Essential Pack/Physics Parameters ---
        self.N_series = tf.constant(self.pack_config.get('num_cells_series', 1), dtype=tf.float32)
        self.Q_pack_nominal_Ah = tf.constant(self.pack_config.get('nominal_capacity_ah', 1.0), dtype=tf.float32)
        self.Q_pack_nominal_Coulombs = self.Q_pack_nominal_Ah * 3600.0

        # --- Loss term weights ---
        self.lambda_data = tf.constant(self.loss_config.get('lambda_data', 1.0), dtype=tf.float32)
        self.lambda_soc_ode = tf.constant(self.loss_config.get('lambda_soc_ode', 0.1), dtype=tf.float32) # Renamed from lambda_pde
        self.lambda_voltage_eqn = tf.constant(self.loss_config.get('lambda_voltage_eqn', 0.1), dtype=tf.float32) # New weight for V equation
        self.lambda_bc_ic = tf.constant(self.loss_config.get('lambda_bc_ic', 0.1), dtype=tf.float32)
        self.lambda_resistance_phys = tf.constant(self.loss_config.get('lambda_resistance_phys', 0.05), dtype=tf.float32) # Penalty for non-physical resistance
        self.lambda_cons = tf.constant(self.loss_config.get('lambda_cons', 0.05), dtype=tf.float32) # Charge conservation
        self.lambda_smooth = tf.constant(self.loss_config.get('lambda_smooth', 0.005), dtype=tf.float32)
        # Removed tau/consistency specific to RC layers/specific states not assumed here

        # --- Load OCV Curve ---
        self.ocv_lookup_path = self.physics_config.get('ocv_soc_lookup_path', None)
        self.soc_ocv_func = self._load_ocv_curve()

        # --- Scaling Factors (Set by Trainer) ---
        self.scale_factors = config.get('training', {}).get('pinn_training', {}).get('scale_factors', {})
        self.time_scale = tf.constant(self.scale_factors.get('time', 1.0), dtype=tf.float32)
        self.voltage_scale = tf.constant(self.scale_factors.get('voltage', 1.0), dtype=tf.float32)
        self.current_scale = tf.constant(self.scale_factors.get('current', 1.0), dtype=tf.float32)
        self.soc_scale = tf.constant(self.scale_factors.get('soc_equiv', 1.0), dtype=tf.float32) # Scale for predicted equivalent SOC
        self.resistance_scale = tf.constant(self.scale_factors.get('resistance_pack', 1.0), dtype=tf.float32) # Scale for predicted pack resistance
        self.temp_scale = tf.constant(self.scale_factors.get('temperature', 1.0), dtype=tf.float32)
        logger.info(f"PhysicsLoss initialized with scales: {self.scale_factors}")


    def _load_ocv_curve(self) -> Optional[Callable]:
        """Loads OCV-SOC curve from CSV and returns an interpolation function."""
        if not self.ocv_lookup_path or not os.path.exists(self.ocv_lookup_path):
            logger.error(f"OCV lookup file not found or path not specified: {self.ocv_lookup_path}")
            return None
        try:
            logger.info(f"Loading OCV curve from: {self.ocv_lookup_path}")
            ocv_df = pd.read_csv(self.ocv_lookup_path)
            # Assume columns are named 'soc' and 'ocv'
            if 'soc' not in ocv_df.columns or 'ocv' not in ocv_df.columns:
                raise ValueError("OCV lookup CSV must contain 'soc' and 'ocv' columns.")
            # Create interpolation function (linear)
            # Ensure data is sorted by SOC
            ocv_df = ocv_df.sort_values(by='soc')
            # Use scipy interp1d for numpy compatibility within tf.py_function later
            interp_func = interp1d(
                ocv_df['soc'].values,
                ocv_df['ocv'].values,
                kind='linear',
                bounds_error=False, # Allow extrapolation using edge values
                fill_value=(ocv_df['ocv'].iloc[0], ocv_df['ocv'].iloc[-1])
            )
            logger.info("OCV curve loaded and interpolation function created.")
            # Wrap with tf.py_function for use in graph mode
            # Need to define output type
            @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
            def ocv_tf_func(soc_tensor):
                # tf.py_function executes eagerly, so use numpy inside
                ocv_val = tf.py_function(
                    func=lambda s: interp_func(s.numpy()).astype(np.float32),
                    inp=[soc_tensor],
                    Tout=tf.float32
                )
                # Set shape if possible (usually output shape is same as input soc_tensor)
                ocv_val.set_shape(soc_tensor.shape)
                return ocv_val

            return ocv_tf_func

        except Exception as e:
            logger.error(f"Failed to load or process OCV curve from {self.ocv_lookup_path}: {e}", exc_info=True)
            return None

    @tf.function
    def _get_ocv(self, soc_equiv_pred_physical: tf.Tensor) -> tf.Tensor:
        """Calculates OCV from SOC using the loaded interpolation function."""
        if self.soc_ocv_func is None:
            logger.error("OCV function not available. Returning zero OCV.")
            return tf.zeros_like(soc_equiv_pred_physical)
        # Ensure SOC is within [0, 1] for lookup stability
        soc_clipped = tf.clip_by_value(soc_equiv_pred_physical, 0.0, 1.0)
        return self.soc_ocv_func(soc_clipped)


    def _get_required_states(self, y_pred_dict: Dict, model_inputs: Dict) -> Optional[Dict]:
        """
        Extracts necessary states and inputs for pack-level physics calculations.
        Performs rescaling from normalized NN outputs to physical units.
        """
        states = {}
        try:
            # --- Predicted States (Normalized) ---
            states['V_pack_pred_norm'] = y_pred_dict['voltage'] # Main output from model structure
            # *** Assumes model predicts these equivalent states ***
            states['soc_avg_equiv_pred_norm'] = y_pred_dict['soc_avg_equiv']
            states['R_pack_effective_pred_norm'] = y_pred_dict['R_pack_effective']
            # Add other predicted states if needed (e.g., eta_equiv_pred_norm)

            # --- Inputs (Normalized, assumed from DataLoader) ---
            states['t_norm'] = model_inputs['time_input'] # Time should be an input feature!
            states['I_pack_norm'] = model_inputs['current_input']
            states['T_pack_norm'] = model_inputs['temperature_input']

            # --- Rescale to Physical Units ---
            states['V_pack_pred'] = states['V_pack_pred_norm'] * self.voltage_scale
            states['soc_avg_equiv_pred'] = states['soc_avg_equiv_pred_norm'] * self.soc_scale
            states['R_pack_effective_pred'] = states['R_pack_effective_pred_norm'] * self.resistance_scale
            states['t'] = states['t_norm'] * self.time_scale
            states['I_pack'] = states['I_pack_norm'] * self.current_scale
            states['T_pack'] = states['T_pack_norm'] * self.temp_scale # Add offset if needed (e.g., Celsius)

            # --- Validation ---
            # Check essential items
            required_keys = ['V_pack_pred', 'soc_avg_equiv_pred', 'R_pack_effective_pred', 't', 'I_pack', 'T_pack']
            missing_keys = [key for key in required_keys if key not in states or states[key] is None]
            if missing_keys:
                 # Try to infer from y_pred_dict and model_inputs directly if keys are different
                 raise ValueError(f"Missing required states/inputs for physics loss: {missing_keys}. Check model output dict and input feature names.")

            return states

        except KeyError as e:
             logger.error(f"Missing key in model output dict ('y_pred_dict') or 'model_inputs': {e}. "
                          f"Ensure PINN model returns '{'voltage'}', '{'soc_avg_equiv'}', '{'R_pack_effective'}' and "
                          f"inputs include '{'time_input'}', '{'current_input'}', '{'temperature_input'}'.")
             return None
        except Exception as e:
             logger.error(f"Error getting required states: {e}", exc_info=True)
             return None


    # --- Physics Equation Residuals (Pack Level) ---

    @tf.function
    def _equivalent_soc_ode_residual(self, soc_equiv_pred: tf.Tensor, dsoc_dt_pred: tf.Tensor, I_pack: tf.Tensor) -> tf.Tensor:
        """Residual for the equivalent average SOC ODE: d(soc)/dt = -I_pack / Q_pack."""
        # d(soc)/dt is provided (calculated using gradients)
        # I_pack should be in Amperes, Q_pack in Coulombs
        # Current convention: Positive for charge, Negative for discharge
        # d(soc)/dt = +I_pack / Q_pack for charge if I_pack is positive charge current
        # d(soc)/dt = -I_pack / Q_pack if I_pack is positive discharge current (check convention)
        # Assuming standard convention: I > 0 charge, I < 0 discharge. SOC increases with charge.
        # d(soc)/dt = I_pack / Q_pack_nominal_Coulombs
        # Residual = dsoc_dt_pred - (I_pack / self.Q_pack_nominal_Coulombs)

        expected_dsoc_dt = I_pack / self.Q_pack_nominal_Coulombs
        residual = dsoc_dt_pred - expected_dsoc_dt
        # Return squared residual for loss calculation
        return tf.square(residual)


    @tf.function
    def _terminal_voltage_residual(self, V_pack_pred: tf.Tensor, soc_equiv_pred: tf.Tensor,
                                   R_pack_effective_pred: tf.Tensor, I_pack: tf.Tensor) -> tf.Tensor:
        """Residual for the pack terminal voltage equation: V_pack = N*OCV(soc_equiv) + I_pack*R_pack_eff."""
        ocv_term = self.N_series * self._get_ocv(soc_equiv_pred) # Calculate OCV based on predicted equiv SOC
        overpotential_term = I_pack * R_pack_effective_pred # Simplified overpotential using effective resistance

        expected_V_pack = ocv_term + overpotential_term
        residual = V_pack_pred - expected_V_pack
        # Return squared residual
        return tf.square(residual)


    # --- Boundary and Initial Conditions ---
    @tf.function
    def _bc_ic_loss(self, states: Dict, y_true_physical: tf.Tensor) -> tf.Tensor:
        """Calculates loss for boundary and initial conditions for the pack."""
        loss = 0.0
        batch_size = tf.shape(states['t'])[0]

        # 1. Initial SOC
        # Assumes the first point in sequence corresponds to initial state
        predicted_initial_soc = states['soc_avg_equiv_pred'][:, 0:1, :] # (Batch, 1, 1)
        target_initial_soc_physical = tf.constant(self.physics_config.get('initial_soc', 1.0), dtype=tf.float32)
        target_initial_soc_tensor = tf.fill(tf.shape(predicted_initial_soc), target_initial_soc_physical)
        loss += tf.reduce_mean(tf.square(predicted_initial_soc - target_initial_soc_tensor))

        # 2. Pack Voltage Limits (Physical)
        pack_voltage_range = self.config.get('data',{}).get('voltage_range', [30.0, 55.0]) # Use pack range from config
        v_pred_physical = states['V_pack_pred'] # Already in physical units
        lower_bound_v = tf.constant(pack_voltage_range[0], dtype=tf.float32)
        upper_bound_v = tf.constant(pack_voltage_range[1], dtype=tf.float32)
        voltage_penalty = tf.reduce_mean(tf.square(tf.nn.relu(lower_bound_v - v_pred_physical)) +
                                         tf.square(tf.nn.relu(v_pred_physical - upper_bound_v)))
        loss += voltage_penalty * 10.0 # Heavier penalty for voltage violation

        # 3. SOC Bounds [0, 1] (Physical)
        soc_pred_physical = states['soc_avg_equiv_pred']
        soc_penalty = tf.reduce_mean(tf.square(tf.nn.relu(0.0 - soc_pred_physical)) +
                                     tf.square(tf.nn.relu(soc_pred_physical - 1.0)))
        loss += soc_penalty

        return loss


    # --- Advanced Physics Terms (Pack Level) ---
    @tf.function
    def _resistance_physicality_penalty(self, states: Dict) -> tf.Tensor:
        """Penalizes non-physical effective pack resistance values."""
        # Resistance should be positive
        R_pack_eff = states['R_pack_effective_pred']
        penalty = tf.reduce_mean(tf.square(tf.nn.relu(-R_pack_eff))) # Penalize negative resistance
        # Could add penalty for excessively large resistance based on physics_config['nominal_pack_resistance_ohm']?
        # penalty += tf.reduce_mean(tf.square(tf.nn.relu(R_pack_eff - max_allowable_R)))
        return penalty


    @tf.function
    def _conservation_laws_penalty(self, states: Dict) -> tf.Tensor:
        """Enforces charge conservation for the pack."""
        loss = 0.0
        # Check: Integral of I_pack dt should match change in charge (Q_pack * delta_SOC)
        if tf.shape(states['t'])[1] > 1: # Need at least two time points
             t_physical = states['t']
             I_pack_physical = states['I_pack']
             soc_equiv_physical = states['soc_avg_equiv_pred']

             delta_t = t_physical[:, 1:, :] - t_physical[:, :-1, :] # (Batch, Time-1, 1)
             # Ensure dt is not zero
             delta_t = tf.where(tf.equal(delta_t, 0.0), 1e-6, delta_t)

             # Integrate current using trapezoidal rule (returns Coulombs)
             integrated_charge_Coulombs = tf.reduce_sum(
                 0.5 * (I_pack_physical[:, 1:, :] + I_pack_physical[:, :-1, :]) * delta_t,
                 axis=1 # Sum over time dimension
             ) # Shape: (Batch, 1)

             # Change in charge based on predicted SOC change
             delta_soc_pred = soc_equiv_physical[:, -1:, :] - soc_equiv_physical[:, 0:1, :] # (Batch, 1, 1)
             delta_charge_from_soc_Coulombs = tf.squeeze(delta_soc_pred, axis=-1) * self.Q_pack_nominal_Coulombs # (Batch, 1)

             # Calculate the residual (difference between integrated current and SOC change)
             conservation_residual = integrated_charge_Coulombs - delta_charge_from_soc_Coulombs
             loss += tf.reduce_mean(tf.square(conservation_residual))

             # Scale the loss? Residual is in Coulombs^2. Maybe scale by Q_pack^2?
             loss = loss / (self.Q_pack_nominal_Coulombs**2 + 1e-9) # Normalize loss


        return loss


    # --- Regularization Terms ---
    @tf.function
    def _smoothness_loss(self, states: Dict) -> tf.Tensor:
        """Penalizes jerky predictions for key pack states."""
        loss = 0.0
        # Penalize large second derivatives of key predictions
        for key in ['V_pack_pred', 'soc_avg_equiv_pred', 'R_pack_effective_pred']:
             if states.get(key) is not None:
                 pred_seq = states[key]
                 if tf.shape(pred_seq)[1] > 2: # Need at least 3 points
                     # Use central difference for second derivative approximation
                     second_diff = pred_seq[:, 2:, :] - 2 * pred_seq[:, 1:-1, :] + pred_seq[:, :-2, :]
                     loss += tf.reduce_mean(tf.square(second_diff))
        return loss


    # --- Main Loss Calculation ---
    # Wrap the main logic in tf.function for potential performance improvement
    # @tf.function
    def call(self, y_true_norm: tf.Tensor, y_pred_struct, **kwargs) -> tf.Tensor:
        """
        Calculates the total physics-informed loss for the battery pack.

        Args:
            y_true_norm (tf.Tensor): Ground truth values (e.g., pack voltage),
                                     normalized as expected by standard loss fns. Shape: (Batch, Time, 1+).
            y_pred_struct : Predicted output from the PINN model (normalized).
                             Expected to be a dictionary containing 'voltage' and
                             other predicted equivalent states (e.g., 'soc_avg_equiv', 'R_pack_effective').
            **kwargs: Must contain 'model_inputs' - a dictionary holding the *normalized* inputs
                      given to the model for this batch, including 'time_input', 'current_input', etc.

        Returns:
            tf.Tensor: The total calculated loss scalar.
        """
        model_inputs_norm = kwargs.get('model_inputs', None)
        if model_inputs_norm is None:
             raise ValueError("Loss function requires 'model_inputs' (normalized) in kwargs.")

        # --- Structure Predictions and Inputs ---
        if not isinstance(y_pred_struct, dict):
             # If model returns tensor directly, assume it's voltage and try to proceed
             # but physics loss will likely fail if other states are needed.
             logger.warning("y_pred_struct is not a dict. Assuming it's 'voltage'. Other physics terms might fail.")
             y_pred_dict_norm = {'voltage': y_pred_struct}
             # Need to provide dummy values for other expected keys if physics needs them? Risky.
             y_pred_dict_norm.setdefault('soc_avg_equiv', tf.zeros_like(y_pred_struct))
             y_pred_dict_norm.setdefault('R_pack_effective', tf.ones_like(y_pred_struct) * 0.01) # Small default resistance
        else:
             y_pred_dict_norm = y_pred_struct

        # --- Get states in physical units ---
        states = self._get_required_states(y_pred_dict_norm, model_inputs_norm)
        if states is None:
            logger.error("Failed to extract/rescale states. Returning high loss.")
            return tf.constant(1e7, dtype=tf.float32) # Return large loss

        # --- Rescale y_true (assuming it's normalized voltage) ---
        # Assumes y_true_norm corresponds to the 'voltage' prediction
        y_true_physical = y_true_norm * self.voltage_scale


        # --- 1. Data Mismatch Loss (Physical Scale) ---
        # Compare predicted pack voltage with ground truth pack voltage
        # Ensure y_true corresponds to the voltage dimension if it contains multiple targets
        loss_data = tf.reduce_mean(tf.square(y_true_physical - states['V_pack_pred']))
        total_loss = self.lambda_data * loss_data


        # --- 2. Physics Residual Losses (Physical Scale) ---
        loss_physics = 0.0

        # Calculate time derivatives using tf.gradient (Recommended approach)
        # Requires the model's call method to be executed within GradientTape context
        # This is typically done in the Trainer's train_step.
        # If grads are passed via kwargs:
        # dsoc_dt_pred = kwargs.get('dsoc_dt_pred', None) # Get pre-computed gradient
        # if dsoc_dt_pred is None: ... fallback ...

        # Fallback/Alternative: Finite difference on predicted sequence (less accurate)
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
             tape.watch(states['t_norm']) # Watch normalized time input
             # Conceptually re-run prediction for SOC based on watched time
             # This needs the model to be structured appropriately or called here
             # Simplified: Assume states['soc_avg_equiv_pred_norm'] is differentiable w.r.t states['t_norm']
             soc_equiv_pred_norm_from_tape = states['soc_avg_equiv_pred_norm'] # Placeholder

        dsoc_dt_norm = tape.gradient(soc_equiv_pred_norm_from_tape, states['t_norm'])

        if dsoc_dt_norm is None:
            logger.warning("Autodiff gradient d(soc_norm)/d(t_norm) is None. Using finite difference approx.")
            # Finite difference approximation on physical values
            if tf.shape(states['t'])[1] > 1:
                 dt_physical = states['t'][:, 1:, :] - states['t'][:, :-1, :]
                 dt_physical = tf.where(tf.equal(dt_physical, 0.0), 1e-6, dt_physical) # Avoid division by zero
                 dsoc_physical = states['soc_avg_equiv_pred'][:, 1:, :] - states['soc_avg_equiv_pred'][:, :-1, :]
                 dsoc_dt_physical_approx = dsoc_physical / dt_physical
                 # Pad to match original sequence length (e.g., repeat first gradient)
                 dsoc_dt_physical = tf.concat([dsoc_dt_physical_approx[:, 0:1, :], dsoc_dt_physical_approx], axis=1)
            else:
                 dsoc_dt_physical = tf.zeros_like(states['soc_avg_equiv_pred']) # Cannot compute diff
        else:
            # Rescale the autodiff gradient computed on normalized values
            dsoc_dt_physical = dsoc_dt_norm * self.soc_scale / self.time_scale
            # Handle potential NaNs from autodiff
            dsoc_dt_physical = tf.where(tf.math.is_nan(dsoc_dt_physical), tf.zeros_like(dsoc_dt_physical), dsoc_dt_physical)


        # a) SOC ODE Residual
        res_soc_ode_sq = self._equivalent_soc_ode_residual(
            states['soc_avg_equiv_pred'], dsoc_dt_physical, states['I_pack']
        )
        loss_physics += self.lambda_soc_ode * tf.reduce_mean(res_soc_ode_sq)


        # b) Terminal Voltage Equation Residual
        res_voltage_sq = self._terminal_voltage_residual(
            states['V_pack_pred'], states['soc_avg_equiv_pred'],
            states['R_pack_effective_pred'], states['I_pack']
        )
        loss_physics += self.lambda_voltage_eqn * tf.reduce_mean(res_voltage_sq)


        # --- 3. Boundary and Initial Condition Loss (Physical Scale) ---
        loss_bc_ic = self._bc_ic_loss(states, y_true_physical)
        loss_physics += self.lambda_bc_ic * loss_bc_ic


        # --- 4. Advanced Physics Penalties (Physical Scale) ---
        # a) Resistance Physicality
        loss_res_phys = self._resistance_physicality_penalty(states)
        loss_physics += self.lambda_resistance_phys * loss_res_phys

        # b) Conservation Law
        loss_cons = self._conservation_laws_penalty(states)
        loss_physics += self.lambda_cons * loss_cons


        # --- 5. Regularization Losses ---
        loss_reg = 0.0
        loss_smooth = self._smoothness_loss(states) # Smoothing on physical states
        loss_reg += self.lambda_smooth * loss_smooth


        # --- Combine Losses ---
        total_loss = loss_data + loss_physics + loss_reg

        # Log individual components for debugging (use tf.print or callbacks)
        # Use tf.summary.scalar in train_step for TensorBoard logging
        # tf.print("Loss - Data:", loss_data, " Phys:", loss_physics, " Reg:", loss_reg)

        # Delete tape if used
        if 'tape' in locals() and tape is not None:
            del tape

        return total_loss

    def get_config(self):
        """Serializes the loss configuration."""
        # Return serializable config parts
        base_config = super().get_config()
        # Reconstruct necessary config parts for from_config
        config_to_save = {
            'config': { # Pass necessary nested configs
                 'training': {'pinn_training': {'loss_config': self.loss_config, 'scale_factors': self.scale_factors}},
                 'physics_config': self.physics_config,
                 'pack_config': self.pack_config,
            },
            'name': self.name
        }
        return config_to_save

    @classmethod
    def from_config(cls, config_dict):
        """Deserializes the loss configuration."""
        # Extract the nested config structure
        nested_config = config_dict.pop('config', {})
        # Cannot reconstruct model reference here
        instance = cls(config=nested_config, name=config_dict.get('name'))
        return instance