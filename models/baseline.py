# battery_project/models/baseline.py

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, List, Optional

# Assuming components are accessible, e.g., from models.components
from .components.attention import TemporalAttention, MultiHeadSelfAttention, FeatureAttention
from .components.builder import build_residual_block
from .components.unified import get_activation, get_normalization, configure_mixed_precision

import logging
logger = logging.getLogger(__name__)


def build_cnn_lstm_model(config: Dict) -> models.Model:
    """Builds a basic CNN-LSTM model based on configuration."""
    # This function remains largely the same, just ensure input shape is correct
    seq_len = config['model']['sequence_length']
    input_features = config['data'].get('features', [])
    if not input_features:
        raise ValueError("Input features list ('config[\"data\"][\"features\"]') cannot be empty.")
    num_features = len(input_features)
    logger.info(f"Building CNN-LSTM model for pack data with Input Shape: (None, {seq_len}, {num_features})")

    cnn_filters = config['model'].get('cnn_filters', [32, 64])
    cnn_kernel_size = config['model'].get('cnn_kernel_size', 3)
    lstm_units = config['model'].get('lstm_units', 50) # Assuming a different key than BiLSTM one
    dropout_rate = config['model'].get('dropout_rate', 0.1)
    final_dense_units = config['model'].get('final_dense_units', [32])

    input_layer = layers.Input(shape=(seq_len, num_features), name="pack_time_series_input")
    x = input_layer

    # CNN layers
    for filters in cnn_filters:
        x = layers.Conv1D(filters=filters, kernel_size=cnn_kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, padding='same')(x) # Optional pooling

    # LSTM layer
    # Need return_sequences=False if it's the last recurrent layer before Dense
    x = layers.LSTM(lstm_units, return_sequences=False, dropout=dropout_rate, name="lstm_layer")(x)

    # Final Dense layers
    for units in final_dense_units:
         x = layers.Dense(units, activation='relu')(x)
         if dropout_rate > 0:
              x = layers.Dropout(dropout_rate)(x)

    # Output layer (assuming single target prediction)
    output_layer = layers.Dense(1, activation='linear', name='target_output')(x) # e.g., pack_soh prediction

    model = models.Model(inputs=input_layer, outputs=output_layer, name="CNN_LSTM_Baseline_Pack")
    model.summary(print_fn=lambda line: logger.info(line))
    return model


def build_cnn_bilstm_attention_model(config: Dict) -> models.Model:
    """
    Builds the CNN-BiLSTM model with Attention mechanism based on configuration.
    Adapted to use pack-level feature dimensions from config.

    Args:
        config (Dict): The main configuration dictionary. Expected keys:
                       config['model']['sequence_length']
                       config['data']['features'] (list of pack feature names)
                       config['model']['cnn_filters'] (list)
                       config['model']['cnn_kernel_size'] (int)
                       config['model']['bilstm_units'] (int)
                       config['model']['attention_units'] (int)
                       config['model']['dropout_rate'] (float)
                       config['model']['final_dense_units'] (list)

    Returns:
        tf.keras.models.Model: The compiled Keras model.
    """
    seq_len = config['model']['sequence_length']
    # Get feature names and count from data config
    input_features = config['data'].get('features', [])
    if not input_features:
        raise ValueError("Input features list ('config[\"data\"][\"features\"]') cannot be empty.")
    num_features = len(input_features)
    logger.info(f"Building CNN-BiLSTM-Attention model for pack data with Input Shape: (None, {seq_len}, {num_features})")
    logger.info(f"Input features expected: {input_features}")

    cnn_filters = config['model'].get('cnn_filters', [32, 64, 64])
    cnn_kernel_size = config['model'].get('cnn_kernel_size', 3)
    bilstm_units = config['model'].get('bilstm_units', 64)
    attention_units = config['model'].get('attention_units', 32)
    dropout_rate = config['model'].get('dropout_rate', 0.2)
    final_dense_units = config['model'].get('final_dense_units', [32])
    use_mha = config['model'].get('use_multihead_attention', False) # Option to use MHA

    # Configure precision policy if needed (usually done globally)
    # configure_mixed_precision(config.get('system', {}).get('mixed_precision', False))

    # --- Input Layer ---
    # Shape determined by config
    input_layer = layers.Input(shape=(seq_len, num_features), name="pack_time_series_input")

    x = input_layer

    # --- CNN Layers ---
    # Use functional API for clarity
    cnn_out = x
    for i, filters in enumerate(cnn_filters):
        cnn_out = layers.Conv1D(
            filters=filters,
            kernel_size=cnn_kernel_size,
            padding='same',
            activation=get_activation(config, 'relu'), # Use unified activation
            name=f'conv1d_{i+1}'
        )(cnn_out)
        cnn_out = get_normalization(config)(name=f'bn_conv_{i+1}')(cnn_out) # Use unified normalization
        # Optional MaxPooling or Strided Conv somewhere? Depends on desired time resolution reduction.
        # cnn_out = layers.MaxPooling1D(pool_size=2, padding='same')(cnn_out)

    # --- BiLSTM Layer ---
    # Needs return_sequences=True for Attention layer
    lstm_out = layers.Bidirectional(
        layers.LSTM(bilstm_units, return_sequences=True), name='bilstm_layer'
    )(cnn_out) # Apply BiLSTM on CNN output features
    if dropout_rate > 0:
        lstm_out = layers.Dropout(dropout_rate, name='bilstm_dropout')(lstm_out)

    # --- Attention Layer ---
    if use_mha:
         # Multi-Head Self-Attention (processes sequence relations)
         # Input shape: (batch, sequence, features)
         attention_out = MultiHeadSelfAttention(
             num_heads=4, # Example, configure if needed
             key_dim=attention_units,
             name='multi_head_attention'
         )(lstm_out) # Query, Value, Key are all lstm_out
         # Need to pool the sequence output from MHA, e.g., GlobalAveragePooling1D
         attention_out = layers.GlobalAveragePooling1D(name='global_avg_pool_attention')(attention_out)
    else:
         # Temporal Attention (weights different time steps)
         attention_context = TemporalAttention(units=attention_units, name='temporal_attention')(lstm_out)
         # attention_context = FeatureAttention(units=attention_units)(lstm_out) # Alternative: Feature attention
         attention_out = attention_context


    # --- Final Dense Layers ---
    dense_out = attention_out
    for i, units in enumerate(final_dense_units):
        dense_out = layers.Dense(
            units,
            activation=get_activation(config, 'relu'),
            name=f'dense_{i+1}'
        )(dense_out)
        dense_out = get_normalization(config)(name=f'bn_dense_{i+1}')(dense_out) # Optional BN
        if dropout_rate > 0:
            dense_out = layers.Dropout(dropout_rate, name=f'dense_dropout_{i+1}')(dense_out)

    # --- Output Layer ---
    # Assuming a single target variable (e.g., pack_soh)
    output_layer = layers.Dense(1, activation='linear', name='target_output')(dense_out)

    # --- Build Model ---
    model = models.Model(inputs=input_layer, outputs=output_layer, name="CNN_BiLSTM_Attention_Pack")

    model.summary(print_fn=lambda line: logger.info(line))

    return model

# --- Example Usage ---
if __name__ == '__main__':
    print("Testing build_cnn_bilstm_attention_model for Pack Data...")
    dummy_config_pack = {
        'data': {
            # Pack features
            'features': ['pack_voltage_norm', 'pack_current_norm', 'pack_temperature_avg_norm',
                         'relative_time_s', 'pack_resistance_est_norm'], # Example 5 features
            'target': ['pack_soh']
        },
        'model': {
            'sequence_length': 60,
            'prediction_length': 1,
            'cnn_filters': [32, 64],
            'cnn_kernel_size': 5,
            'bilstm_units': 50,
            'attention_units': 25,
            'dropout_rate': 0.15,
            'final_dense_units': [20],
            'use_multihead_attention': False,
            'activation': 'relu', # For get_activation
            'normalization': 'batch', # For get_normalization
        },
        'system': {'mixed_precision': False}
    }

    # Build the model
    baseline_model = build_cnn_bilstm_attention_model(dummy_config_pack)

    # Check input and output shapes
    batch_size = 8
    seq_len = dummy_config_pack['model']['sequence_length']
    num_features = len(dummy_config_pack['data']['features'])
    dummy_input = tf.random.normal((batch_size, seq_len, num_features))
    output = baseline_model(dummy_input)

    print("\n--- Model Input/Output Shapes ---")
    print(f"Input shape check: {baseline_model.input_shape} == {(None, seq_len, num_features)}")
    print(f"Output shape check: {baseline_model.output_shape} == {(None, 1)}")

    assert baseline_model.input_shape == (None, seq_len, num_features)
    assert baseline_model.output_shape == (None, 1)

    print("\nbuild_cnn_bilstm_attention_model test completed successfully.")