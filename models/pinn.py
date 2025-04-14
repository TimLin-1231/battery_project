# battery_project/models/pinn.py

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, List, Optional, Tuple

# Assuming components are accessible, e.g., from models.components
# from .components.rcparams import MultiRCParams # Keep commented unless re-integrated
from .components.attention import TemporalAttention, MultiHeadSelfAttention # Keep if using attention
from .components.builder import build_residual_block # Keep if using residual blocks
from .components.unified import get_activation, get_normalization, configure_mixed_precision


import logging
logger = logging.getLogger(__name__)


def build_pinn_pack_model(config: Dict) -> models.Model:
    """
    Builds a Physics-Informed Neural Network (PINN) model suitable for
    battery pack modeling based on an Equivalent Single Cell Model (ESCM) approach.

    This model processes pack-level time-series inputs and predicts normalized pack voltage
    along with required equivalent internal states (e.g., SOC_equiv, R_pack_effective)
    needed by the EnhancedPhysicsLoss function.

    Args:
        config (Dict): The main configuration dictionary. Expected keys:
                       config['model']['sequence_length']
                       config['data']['features'] (list of pack feature names)
                       config['model']['pinn']['lstm_units']
                       config['model']['dropout_rate'] (optional)
                       config['model']['pinn']['use_bilstm'] (optional, boolean)
                       config['model']['pinn']['output_structure'] (should be 'dict')

    Returns:
        tf.keras.models.Model: The compiled Keras model.
    """
    seq_len = config['model']['sequence_length']
    # Determine number of input features from data config
    input_features = config['data'].get('features', [])
    if not input_features:
        raise ValueError("Input features list ('config[\"data\"][\"features\"]') cannot be empty.")
    num_features = len(input_features)
    logger.info(f"Building PINN Pack model with Input Shape: (None, {seq_len}, {num_features})")
    logger.info(f"Input features expected: {input_features}")

    lstm_units = config['model'].get('pinn', {}).get('lstm_units', 64)
    use_bilstm = config['model'].get('pinn', {}).get('use_bilstm', True)
    dropout_rate = config['model'].get('dropout_rate', 0.1)
    output_structure = config['model'].get('pinn', {}).get('output_structure', 'dict')
    if output_structure != 'dict':
        logger.warning(f"PINN output_structure is '{output_structure}', but loss expects 'dict'. Adjusting.")
        # Force output structure to dict as required by the loss function
        output_structure = 'dict'

    # Configure precision policy
    # configure_mixed_precision(config.get('system', {}).get('mixed_precision', False)) # Done globally now?

    # --- Input Layer ---
    input_layer = layers.Input(shape=(seq_len, num_features), name="pack_time_series_input")

    # --- Optional: Initial Feature Processing (e.g., CNN) ---
    # x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
    # x = layers.BatchNormalization()(x)
    x = input_layer # Start directly with LSTM/BiLSTM if no CNN

    # --- Recurrent Layers for Temporal Dynamics ---
    recurrent_layer = layers.LSTM
    if use_bilstm:
        recurrent_layer = layers.Bidirectional
        logger.debug("Using Bidirectional LSTM layer.")
        lstm_layer = recurrent_layer(layers.LSTM(lstm_units, return_sequences=True), name="bilstm_layer")
    else:
         logger.debug("Using standard LSTM layer.")
         lstm_layer = recurrent_layer(lstm_units, return_sequences=True, name="lstm_layer")

    x = lstm_layer(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="lstm_dropout")(x)

    # Add another LSTM layer?
    # x = layers.LSTM(lstm_units // 2, return_sequences=True, name="lstm_layer_2")(x)
    # if dropout_rate > 0:
    #     x = layers.Dropout(dropout_rate, name="lstm_dropout_2")(x)

    # --- Output Heads for Required States (TimeDistributed Dense) ---
    # These layers predict the normalized values required by the loss function.

    # 1. Voltage Prediction Head (Main Output)
    voltage_output = layers.TimeDistributed(
        layers.Dense(1, activation='linear'), name='voltage_norm_output'
    )(x)

    # 2. Equivalent SOC Prediction Head
    # Use sigmoid activation assuming SOC is scaled between 0 and 1
    soc_equiv_output = layers.TimeDistributed(
        layers.Dense(1, activation='sigmoid'), name='soc_avg_equiv_norm_output'
    )(x)

    # 3. Effective Pack Resistance Prediction Head
    # Use softplus activation to ensure predicted resistance is non-negative after rescaling
    r_pack_eff_output = layers.TimeDistributed(
        layers.Dense(1, activation='softplus'), name='R_pack_effective_norm_output'
    )(x)

    # --- TODO (Optional): Add other predicted states if needed ---
    # Example: Equivalent Overpotential Head
    # eta_equiv_output = layers.TimeDistributed(
    #     layers.Dense(1, activation='linear'), name='eta_equiv_norm_output'
    # )(x)


    # --- Assemble Output Dictionary ---
    # The keys MUST match those expected by EnhancedPhysicsLoss._get_required_states
    output_dict = {
        'voltage': voltage_output,
        'soc_avg_equiv': soc_equiv_output,
        'R_pack_effective': r_pack_eff_output,
        # Add other predicted states here if implemented
        # 'eta_equiv': eta_equiv_output,
    }
    logger.info(f"Model outputs (normalized): {list(output_dict.keys())}")

    # --- Build the Model ---
    model = models.Model(inputs=input_layer, outputs=output_dict, name="PINN_Pack_ESCM_Model")

    # --- Log Model Summary ---
    # Use a lambda to capture the summary string in the logger
    model.summary(print_fn=lambda line: logger.info(line))

    return model


# --- Example Usage ---
if __name__ == '__main__':
    print("Testing build_pinn_pack_model...")
    dummy_config = {
        'data': {
            # Pack features
            'features': ['pack_voltage_norm', 'pack_current_norm', 'pack_temperature_avg_norm', 'time_input_norm'],
            'target': ['pack_voltage_norm'] # Target usually used for data loss part
        },
        'model': {
            'sequence_length': 50,
            'prediction_length': 1, # Not directly used by model build, but good for context
            'dropout_rate': 0.1,
            'pinn': {
                'lstm_units': 40,
                'use_bilstm': True,
                'output_structure': 'dict',
                 # 'state_dims' is less relevant now as outputs are explicit heads
                 'state_dims': {'soc_avg_equiv': 1, 'R_pack_effective': 1},
            }
        },
         'system': {'mixed_precision': False} # Example system config
    }

    # Build the model
    pinn_model = build_pinn_pack_model(dummy_config)

    # Check output structure and names
    assert isinstance(pinn_model.output, dict)
    assert 'voltage' in pinn_model.output
    assert 'soc_avg_equiv' in pinn_model.output
    assert 'R_pack_effective' in pinn_model.output

    # Check output shapes (assuming batch size = None)
    batch_size = 4
    seq_len = dummy_config['model']['sequence_length']
    num_features = len(dummy_config['data']['features'])
    dummy_input = tf.random.normal((batch_size, seq_len, num_features))
    outputs = pinn_model(dummy_input)

    print("\n--- Model Output Shapes ---")
    for name, tensor in outputs.items():
        print(f"'{name}': {tensor.shape}")
        if name == 'voltage':
             assert tensor.shape == (batch_size, seq_len, 1)
        elif name == 'soc_avg_equiv':
             assert tensor.shape == (batch_size, seq_len, 1)
        elif name == 'R_pack_effective':
             assert tensor.shape == (batch_size, seq_len, 1)

    print("\nbuild_pinn_pack_model test completed successfully.")