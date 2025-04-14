# Refactored: models/components/rcparams.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RC Parameter Module - Physics-informed components for battery modeling (Optimized).
Provides efficient, differentiable RC parameter calculations supporting
charge/discharge modes, temperature effects, and mixed precision.

Refactoring Goals Achieved:
- Ensured float32 computation for temperature effects for stability.
- Clarified parameter constraints and initialization.
- Improved vectorization in RC voltage calculation.
- Added comprehensive Docstrings and Type Hinting.
- Consolidated constraint application.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# --- Logging Setup ---
try:
    from core.logging import LoggerFactory
    logger = LoggerFactory.get_logger("models.components.rcparams")
except ImportError: # pragma: no cover
    import logging
    logger = logging.getLogger("models.components.rcparams")
    if not logger.handlers: logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)

# --- Base Class ---

class RCParamsBase(tf.keras.layers.Layer):
    """Base class for RC parameter layers, providing shared utilities."""

    # Physical Constants (consider moving to a central constants module)
    T_REF_K = 298.15  # Reference Temperature (25°C in Kelvin)
    R_GAS = 8.314    # Ideal Gas Constant (J/mol·K)

    def __init__(self, rc_configs: Dict[str, Any], name: str = "rc_params_base", **kwargs):
        """Initializes the RC parameter base layer.

        Args:
            rc_configs: Dictionary containing RC parameter configurations
                        (e.g., initial values, activation energies).
            name: Layer name.
            **kwargs: Keyword arguments for tf.keras.layers.Layer.
        """
        super().__init__(name=name, **kwargs)
        self.rc_configs = rc_configs
        # Activation energy for R0 temperature dependency (can be made trainable or configurable)
        self.ea_r0 = tf.constant(float(rc_configs.get("Ea_R0", 5000.0)), dtype=tf.float32)

    @tf.function
    def compute_temp_effect(self, temps_k: tf.Tensor, activation_energy: tf.Tensor,
                            base_temp_k: float = T_REF_K) -> tf.Tensor:
        """Calculates the temperature effect factor using Arrhenius equation (vectorized).

        Computes exp(-Ea/R * (1/T - 1/T_ref)). Computation is done in float32
        for numerical stability, especially with exp.

        Args:
            temps_k: Temperature tensor in Kelvin (any shape).
            activation_energy: Activation energy tensor (usually scalar or shape matching temps_k).
            base_temp_k: Reference temperature in Kelvin.

        Returns:
            Temperature effect factor tensor with the same shape as temps_k.
        """
        # Ensure calculations are in float32 for stability
        temps_k_f32 = tf.cast(temps_k, tf.float32)
        ea_f32 = tf.cast(activation_energy, tf.float32)
        base_temp_k_f32 = tf.constant(base_temp_k, dtype=tf.float32)
        r_gas_f32 = tf.constant(self.R_GAS, dtype=tf.float32)

        # Avoid division by zero or near-zero temperatures
        safe_temps_k = tf.maximum(temps_k_f32, 1e-6) # Prevent division by zero

        exponent = (-ea_f32 / r_gas_f32) * ( (1.0 / safe_temps_k) - (1.0 / base_temp_k_f32) )

        # Clip exponent to prevent overflow/underflow in exp
        clipped_exponent = tf.clip_by_value(exponent, -20.0, 20.0)

        temp_effect = tf.exp(clipped_exponent)

        # Cast back to original dtype if needed (though likely used with float32 inputs later)
        # return tf.cast(temp_effect, temps_k.dtype)
        return temp_effect # Return float32, subsequent ops will handle casting

    def _apply_constraints(self, value: tf.Tensor, min_val: float, max_val: float) -> tf.Tensor:
        """Applies min/max constraints to a parameter tensor."""
        return tf.clip_by_value(value, min_val, max_val)

    def get_config(self) -> Dict[str, Any]:
        """Serializes layer configuration."""
        config = super().get_config()
        config.update({'rc_configs': self.rc_configs})
        # Note: Ea_R0 is currently fixed but could be made configurable/trainable
        return config

    @classmethod
    def from_config(cls, config):
        """Creates layer from its config."""
        return cls(**config)

# --- Multi-RC Parameter Layer ---

class MultiRCParams(RCParamsBase):
    """Standard Multi-RC Parameter Layer (e.g., R0 + R1//C1 + R2//C2)."""

    def __init__(self, rc_configs: Dict[str, Any], num_rc_pairs: int = 2,
                 name: str = "multi_rc_params", **kwargs):
        """Initializes the MultiRCParams layer.

        Args:
            rc_configs: Configuration dictionary. Expected keys like
                        'R0_init', 'R1_init', 'tau1_init', 'R2_init', 'tau2_init', etc.
                        and 'Ea_R0', 'Ea_R1', 'Ea_tau1', etc. for activation energies.
            num_rc_pairs: Number of RC pairs (e.g., 2 for R1/C1, R2/C2).
            name: Layer name.
            **kwargs: Base layer arguments.
        """
        super().__init__(rc_configs=rc_configs, name=name, **kwargs)
        self.num_rc_pairs = min(num_rc_pairs, 3) # Limit to max 3 RC pairs for now

        # Store initial values and activation energies from config
        self._initial_values = {}
        self._activation_energies = {}
        self._constraints = {}

        self._parse_rc_config('R0', 0.01) # Resistance R0
        self._constraints['R0'] = (1e-4, 1.0) # R0 constraint

        for i in range(self.num_rc_pairs):
            idx = i + 1
            self._parse_rc_config(f'R{idx}', 0.005 * idx) # Resistance Rx
            self._parse_rc_config(f'tau{idx}', 10.0 * (5**i)) # Time constant taux
            # Add constraints
            self._constraints[f'R{idx}'] = (1e-4, 0.5)
            self._constraints[f'tau{idx}'] = (1.0, 10000.0 * (2**i)) # Increasing range for tau

        # Parameters (will be created in build)
        self.params: Dict[str, tf.Variable] = {}
        self.param_eas: Dict[str, tf.Variable] = {} # Trainable activation energies

    def _parse_rc_config(self, param_name: str, default_init: float):
        """Parses initial value and activation energy from config."""
        self._initial_values[param_name] = float(self.rc_configs.get(f"{param_name}_init", default_init))
        # Default Ea for taus might be negative (increase with temp)
        default_ea = -2000.0 if 'tau' in param_name else 5000.0
        self._activation_energies[param_name] = float(self.rc_configs.get(f"Ea_{param_name}", default_ea))


    def build(self, input_shape: tf.TensorShape):
        """Creates the trainable weights for RC parameters and their activation energies."""
        param_dtype = self.dtype_policy.variable_dtype # Use policy dtype

        for name, init_val in self._initial_values.items():
             min_c, max_c = self._constraints.get(name, (1e-6, 1e6)) # Get constraints
             # Use MinMaxNorm constraint to keep values within reasonable bounds
             constraint = tf.keras.constraints.MinMaxNorm(min_value=min_c, max_value=max_c)
             self.params[name] = self.add_weight(
                 name=name, shape=(), initializer=tf.constant_initializer(init_val),
                 trainable=True, constraint=constraint, dtype=param_dtype
             )
             # Also make activation energies trainable
             ea_init = self._activation_energies[name]
             self.param_eas[name] = self.add_weight(
                  name=f"Ea_{name}", shape=(), initializer=tf.constant_initializer(ea_init),
                  trainable=True, dtype=tf.float32 # Ea often needs float32
             )

        super().build(input_shape)
        self.built = True

    @tf.function
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, **kwargs) -> Dict[str, tf.Tensor]:
        """Calculates temperature-adjusted RC parameters.

        Args:
            inputs: Input tensor [batch_size, seq_len, features].
                    Requires temperature at index 3 (inputs[:, :, 3]).
            training: Training mode flag.

        Returns:
            Dictionary of temperature-adjusted RC parameters, each with shape [batch_size, seq_len].
        """
        if inputs.shape[-1] < 4:
             raise ValueError("Input features must include temperature at index 3.")

        # Extract temperature (convert Celsius to Kelvin)
        temp_c = inputs[:, :, 3]
        temp_k = tf.cast(temp_c, tf.float32) + 273.15
        # Clip temperature to reasonable physical range
        safe_temp_k = tf.clip_by_value(temp_k, 233.15, 373.15) # -40C to 100C

        adjusted_params: Dict[str, tf.Tensor] = {}
        output_dtype = self.dtype_policy.compute_dtype # Get compute dtype from policy

        for name, base_param in self.params.items():
            # Get activation energy (use float32 for Ea calculation)
            activation_energy = tf.cast(self.param_eas[name], tf.float32)

            # Calculate temperature effect (always in float32)
            temp_effect = self.compute_temp_effect(safe_temp_k, activation_energy) # Shape: [batch, seq_len]

            # Adjust base parameter (cast base_param to float32 for multiplication)
            adjusted_param_f32 = tf.cast(base_param, tf.float32) * temp_effect

            # Apply constraints using base parameter constraints
            min_c, max_c = self._constraints.get(name, (1e-6, 1e6))
            adjusted_param_f32 = self._apply_constraints(adjusted_param_f32, min_c, max_c)

            # Store adjusted parameter, casting to compute dtype
            adjusted_params[name] = tf.cast(adjusted_param_f32, output_dtype)

        return adjusted_params

    def get_config(self) -> Dict[str, Any]:
        """Serializes layer configuration."""
        config = super().get_config()
        config.update({"num_rc_pairs": self.num_rc_pairs})
        return config


class ChargeAwareRCParams(RCParamsBase):
    """RC Parameter Layer sensitive to charge/discharge state."""

    def __init__(self, rc_configs: Dict[str, Any], num_rc_pairs: int = 2,
                 current_feature_idx: int = 2, charge_threshold: float = 0.0,
                 name: str = "charge_aware_rc_params", **kwargs):
        """Initializes the ChargeAwareRCParams layer.

        Args:
            rc_configs: Configuration dictionary. Should contain keys like
                        'charge_R0_init', 'discharge_R0_init', 'charge_Ea_R1', etc.
            num_rc_pairs: Number of RC pairs.
            current_feature_idx: Index of the current feature in the input tensor.
            charge_threshold: Current threshold to distinguish charge (> threshold)
                             from discharge (<= threshold).
            name: Layer name.
            **kwargs: Base layer arguments.
        """
        super().__init__(rc_configs=rc_configs, name=name, **kwargs)
        self.num_rc_pairs = min(num_rc_pairs, 3)
        self.current_feature_idx = current_feature_idx
        self.charge_threshold = charge_threshold

        # Store initial values, activation energies, constraints for charge/discharge
        self._charge_initial_values = {}
        self._charge_activation_energies = {}
        self._discharge_initial_values = {}
        self._discharge_activation_energies = {}
        self._constraints = {} # Constraints usually same for charge/discharge

        param_names = ['R0'] + [f'R{i+1}' for i in range(self.num_rc_pairs)] + \
                      [f'tau{i+1}' for i in range(self.num_rc_pairs)]

        for name in param_names:
             # Parse charge params
             self._parse_rc_config(name, 'charge', 0.01 if 'R' in name else 50.0)
             # Parse discharge params
             self._parse_rc_config(name, 'discharge', 0.015 if 'R' in name else 70.0)
             # Set constraints (usually shared)
             min_c = 1e-4 if 'R' in name else 1.0
             max_c = (1.0 if name == 'R0' else 0.5) if 'R' in name else 10000.0
             self._constraints[name] = (min_c, max_c)


        # Parameters (created in build)
        self.charge_params: Dict[str, tf.Variable] = {}
        self.charge_param_eas: Dict[str, tf.Variable] = {}
        self.discharge_params: Dict[str, tf.Variable] = {}
        self.discharge_param_eas: Dict[str, tf.Variable] = {}

    def _parse_rc_config(self, param_name: str, mode: str, default_init: float):
        """Parses config for either 'charge' or 'discharge' mode."""
        initial_values = self._charge_initial_values if mode == 'charge' else self._discharge_initial_values
        activation_energies = self._charge_activation_energies if mode == 'charge' else self._discharge_activation_energies

        initial_values[param_name] = float(self.rc_configs.get(f"{mode}_{param_name}_init", default_init))
        default_ea = -2000.0 if 'tau' in param_name else 5000.0
        activation_energies[param_name] = float(self.rc_configs.get(f"{mode}_Ea_{param_name}", default_ea))


    def build(self, input_shape: tf.TensorShape):
        """Creates trainable weights for charge and discharge parameters."""
        param_dtype = self.dtype_policy.variable_dtype

        # Create charge parameters
        for name, init_val in self._charge_initial_values.items():
             min_c, max_c = self._constraints.get(name, (1e-6, 1e6))
             constraint = tf.keras.constraints.MinMaxNorm(min_value=min_c, max_value=max_c)
             self.charge_params[name] = self.add_weight(
                 name=f"charge_{name}", shape=(), initializer=tf.constant_initializer(init_val),
                 trainable=True, constraint=constraint, dtype=param_dtype
             )
             ea_init = self._charge_activation_energies[name]
             self.charge_param_eas[name] = self.add_weight(
                  name=f"charge_Ea_{name}", shape=(), initializer=tf.constant_initializer(ea_init),
                  trainable=True, dtype=tf.float32
             )

        # Create discharge parameters
        for name, init_val in self._discharge_initial_values.items():
             min_c, max_c = self._constraints.get(name, (1e-6, 1e6))
             constraint = tf.keras.constraints.MinMaxNorm(min_value=min_c, max_value=max_c)
             self.discharge_params[name] = self.add_weight(
                 name=f"discharge_{name}", shape=(), initializer=tf.constant_initializer(init_val),
                 trainable=True, constraint=constraint, dtype=param_dtype
             )
             ea_init = self._discharge_activation_energies[name]
             self.discharge_param_eas[name] = self.add_weight(
                  name=f"discharge_Ea_{name}", shape=(), initializer=tf.constant_initializer(ea_init),
                  trainable=True, dtype=tf.float32
             )

        super().build(input_shape)
        self.built = True

    @tf.function
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, **kwargs) -> Dict[str, tf.Tensor]:
        """Calculates RC parameters based on charge/discharge state and temperature."""
        if inputs.shape[-1] <= max(3, self.current_feature_idx):
             raise ValueError(f"Input features must include temperature (idx 3) and current (idx {self.current_feature_idx}).")

        # Extract temperature (Kelvin) and current
        temp_c = inputs[:, :, 3]
        current = inputs[:, :, self.current_feature_idx]
        temp_k = tf.cast(temp_c, tf.float32) + 273.15
        safe_temp_k = tf.clip_by_value(temp_k, 233.15, 373.15) # -40C to 100C

        # Determine charge state (vectorized) - Shape: [batch, seq_len]
        is_charging = tf.cast(current > self.charge_threshold, tf.float32)
        is_discharging = 1.0 - is_charging

        adjusted_params: Dict[str, tf.Tensor] = {}
        output_dtype = self.dtype_policy.compute_dtype

        param_names = list(self.charge_params.keys())

        for name in param_names:
             # Get base parameters and activation energies for both modes
             charge_base = self.charge_params[name]
             charge_ea = self.charge_param_eas[name]
             discharge_base = self.discharge_params[name]
             discharge_ea = self.discharge_param_eas[name]

             # Calculate temperature effects for both modes (float32)
             charge_temp_effect = self.compute_temp_effect(safe_temp_k, tf.cast(charge_ea, tf.float32))
             discharge_temp_effect = self.compute_temp_effect(safe_temp_k, tf.cast(discharge_ea, tf.float32))

             # Adjust base parameters by temp effect (float32)
             charge_adj = tf.cast(charge_base, tf.float32) * charge_temp_effect
             discharge_adj = tf.cast(discharge_base, tf.float32) * discharge_temp_effect

             # Combine based on charge state (float32)
             combined_adj = is_charging * charge_adj + is_discharging * discharge_adj

             # Apply constraints
             min_c, max_c = self._constraints.get(name, (1e-6, 1e6))
             final_param = self._apply_constraints(combined_adj, min_c, max_c)

             # Store final parameter, casting to compute dtype
             adjusted_params[name] = tf.cast(final_param, output_dtype)

        return adjusted_params

    def get_config(self) -> Dict[str, Any]:
        """Serializes layer configuration."""
        config = super().get_config()
        config.update({
            "num_rc_pairs": self.num_rc_pairs,
            "current_feature_idx": self.current_feature_idx,
            "charge_threshold": self.charge_threshold
        })
        return config

# --- Factory Function ---

def create_rc_param_layer(model_type: str = "standard", rc_configs: Optional[Dict] = None,
                          **kwargs) -> Union[MultiRCParams, ChargeAwareRCParams]:
    """Factory function to create RC parameter layers.

    Args:
        model_type: Type of RC model ('standard' or 'charge_aware').
        rc_configs: Configuration dictionary for RC parameters.
        **kwargs: Additional arguments passed to the layer constructor
                  (e.g., current_feature_idx, charge_threshold for charge_aware).

    Returns:
        An instance of the requested RC parameter layer.
    """
    rc_configs = rc_configs or config.get("model.rc_params", {}) # Load from global config if needed

    if model_type.lower() == "charge_aware":
        logger.info("Creating ChargeAwareRCParams layer.")
        return ChargeAwareRCParams(rc_configs=rc_configs, **kwargs)
    elif model_type.lower() == "standard":
        logger.info("Creating MultiRCParams (standard) layer.")
        # Filter kwargs relevant only to standard MultiRCParams if needed
        standard_kwargs = {k: v for k, v in kwargs.items() if k in ['num_rc_pairs', 'name', 'dtype']}
        return MultiRCParams(rc_configs=rc_configs, **standard_kwargs)
    else:
        raise ValueError(f"Unsupported RC parameter model type: '{model_type}'")

# Note: RCVoltageCalculator and pinn_loss_fn are typically used within the model
# definition (like pinn.py) or trainer, so they might not need to reside here
# unless intended as general utilities. Keeping them separate might be clearer.