# battery_project/models/gan.py

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, List, Optional

import logging
logger = logging.getLogger(__name__)

# --- RCGAN Generator ---
def build_generator(config: Dict) -> models.Model:
    """
    Builds the Generator model for the Recurrent Conditional GAN (RCGAN).
    Adapted for generating battery pack time-series data.

    Args:
        config (Dict): The main configuration dictionary. Expected keys:
                       config['model']['sequence_length']
                       config['model']['gan']['generator']['lstm_units']
                       config['model']['gan']['generator']['noise_dim']
                       config['model']['gan']['generator']['conditional_dim']
                       config['data']['augmentation']['gan']['output_features'] (list of feature names)

    Returns:
        tf.keras.models.Model: The Generator model.
    """
    seq_len = config['model']['sequence_length']
    gen_config = config['model'].get('gan', {}).get('generator', {})
    lstm_units = gen_config.get('lstm_units', 128)
    noise_dim = gen_config.get('noise_dim', 100)
    conditional_dim = gen_config.get('conditional_dim', 1) # Dimension of condition (e.g., target SOH)

    # Get the *output* dimension based on the features the GAN should generate
    gan_output_features = config['data'].get('augmentation', {}).get('gan', {}).get('output_features', [])
    if not gan_output_features:
        raise ValueError("GAN 'output_features' list must be specified in config['data']['augmentation']['gan']")
    output_dim = len(gan_output_features)
    logger.info(f"Building RCGAN Generator. Input noise={noise_dim}, condition={conditional_dim}. Output dim={output_dim} features for seq_len={seq_len}.")
    logger.info(f"Generator expected output features: {gan_output_features}")


    # Combined input dimension for noise and condition
    generator_input_dim = noise_dim + conditional_dim
    input_layer = layers.Input(shape=(generator_input_dim,), name="generator_noise_condition_input")

    # Conditioning Method A: Dense -> Repeat -> LSTM initial state (or input)
    # Project the combined input to match LSTM state size or desired intermediate dim
    x = layers.Dense(lstm_units * seq_len, activation='relu')(input_layer) # Expand to cover seq_len
    x = layers.Reshape((seq_len, lstm_units))(x) # Reshape into sequence format
    # Alternative: Use RepeatVector if input feeds each step
    # x = layers.Dense(lstm_units, activation='relu')(input_layer)
    # x = layers.RepeatVector(seq_len)(x)

    # LSTM layers to generate sequence dynamics
    # Use return_sequences=True for all recurrent layers except possibly the last if feeding Dense directly
    x = layers.LSTM(lstm_units, return_sequences=True, name="generator_lstm_1")(x)
    # x = layers.LSTM(lstm_units, return_sequences=True, name="generator_lstm_2")(x) # Optional second layer

    # Output layer: TimeDistributed Dense to generate features for each time step
    # Output dimension must match the number of pack features to generate
    output_layer = layers.TimeDistributed(
        layers.Dense(output_dim, activation='tanh'), # Use tanh common for GANs (-1 to 1), data should be scaled accordingly
        name='generator_pack_feature_output'
    )(x)

    model = models.Model(inputs=input_layer, outputs=output_layer, name="RCGAN_Generator_Pack")
    model.summary(print_fn=lambda line: logger.info(line))
    return model


# --- RCGAN Discriminator ---
def build_discriminator(config: Dict) -> models.Model:
    """
    Builds the Discriminator model for the Recurrent Conditional GAN (RCGAN).
    Adapted for discriminating real/fake battery pack time-series data.

    Args:
        config (Dict): The main configuration dictionary. Expected keys:
                       config['model']['sequence_length']
                       config['data']['features'] (list of real pack feature names)
                       config['model']['gan']['discriminator']['cnn_filters'] (list)
                       config['model']['gan']['discriminator']['cnn_kernel_size'] (int)
                       config['model']['gan']['discriminator']['lstm_units'] (int)
                       config['model']['gan']['discriminator']['conditional_dim'] (int)
                       config['model']['gan']['discriminator']['dropout_rate'] (float)

    Returns:
        tf.keras.models.Model: The Discriminator model.
    """
    seq_len = config['model']['sequence_length']
    # Discriminator input feature count MUST match the real data features
    real_features = config['data'].get('features', [])
    if not real_features:
         raise ValueError("Discriminator needs 'features' list in config['data'] to determine input dimension.")
    num_features = len(real_features) # Input dimension is number of real pack features

    disc_config = config['model'].get('gan', {}).get('discriminator', {})
    cnn_filters = disc_config.get('cnn_filters', [64, 128])
    cnn_kernel_size = disc_config.get('cnn_kernel_size', 5)
    lstm_units = disc_config.get('lstm_units', 64)
    conditional_dim = disc_config.get('conditional_dim', 1) # Should match generator's conditional_dim
    dropout_rate = disc_config.get('dropout_rate', 0.3)

    logger.info(f"Building RCGAN Discriminator. Input seq_len={seq_len}, features={num_features}, condition={conditional_dim}.")
    logger.info(f"Discriminator expects real features: {real_features}")

    # Input for the time series sequence (real or fake pack data)
    sequence_input = layers.Input(shape=(seq_len, num_features), name="discriminator_sequence_input")

    # Input for the condition
    condition_input = layers.Input(shape=(conditional_dim,), name="discriminator_condition_input")

    # --- Process Sequence (CNN -> LSTM) ---
    seq_processed = sequence_input
    # Optional CNN layers for feature extraction along time
    for i, filters in enumerate(cnn_filters):
         seq_processed = layers.Conv1D(
             filters=filters,
             kernel_size=cnn_kernel_size,
             strides=2, # Downsample time dimension
             padding='same',
             activation='relu', # Using relu, consider LeakyReLU often used in GANs
             name=f'discriminator_conv1d_{i+1}'
         )(seq_processed)
         if dropout_rate > 0:
              seq_processed = layers.Dropout(dropout_rate)(seq_processed)

    # LSTM layer to capture temporal dependencies
    # return_sequences=False as we want a single vector representing the sequence
    seq_features = layers.LSTM(lstm_units, return_sequences=False, name="discriminator_lstm")(seq_processed)

    # --- Process Condition ---
    # Embed the condition or use it directly
    condition_features = layers.Dense(lstm_units // 2, activation='relu', name="condition_embedding")(condition_input) # Project condition

    # --- Combine Sequence and Condition Features ---
    combined_features = layers.Concatenate()([seq_features, condition_features])

    # --- Final Dense Layers for Classification ---
    x = layers.Dense(64, activation='relu')(combined_features) # Using relu, consider LeakyReLU
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    # Output layer: Single neuron for real/fake prediction (no activation for WGAN-GP)
    output_layer = layers.Dense(1, name='discriminator_output')(x)

    model = models.Model(inputs=[sequence_input, condition_input], outputs=output_layer, name="RCGAN_Discriminator_Pack")
    model.summary(print_fn=lambda line: logger.info(line))
    return model


# --- Example Usage ---
if __name__ == '__main__':
    print("Testing RCGAN Generator and Discriminator build for Pack Data...")
    dummy_config_gan = {
        'data': {
            'features': ['pack_voltage_norm', 'pack_current_norm', 'pack_temperature_avg_norm', 'relative_time_s'], # Real features
            'augmentation': {
                'gan': {
                     # Features GAN should output (MUST match data.features for standard GAN)
                    'output_features': ['pack_voltage_norm', 'pack_current_norm', 'pack_temperature_avg_norm', 'relative_time_s']
                }
            }
        },
        'model': {
            'sequence_length': 50,
            'gan': {
                'generator': {
                    'lstm_units': 64,
                    'noise_dim': 50,
                    'conditional_dim': 1
                },
                'discriminator': {
                    'cnn_filters': [32, 64],
                    'cnn_kernel_size': 3,
                    'lstm_units': 32,
                    'conditional_dim': 1,
                    'dropout_rate': 0.2
                }
            }
        }
    }

    # --- Build Generator ---
    print("\n--- Building Generator ---")
    generator = build_generator(dummy_config_gan)
    gen_output_dim = len(dummy_config_gan['data']['augmentation']['gan']['output_features'])
    assert generator.output_shape == (None, dummy_config_gan['model']['sequence_length'], gen_output_dim)
    assert generator.input_shape == (None, dummy_config_gan['model']['gan']['generator']['noise_dim'] + dummy_config_gan['model']['gan']['generator']['conditional_dim'])

    # --- Build Discriminator ---
    print("\n--- Building Discriminator ---")
    discriminator = build_discriminator(dummy_config_gan)
    disc_input_features = len(dummy_config_gan['data']['features'])
    assert discriminator.input_shape == [(None, dummy_config_gan['model']['sequence_length'], disc_input_features), (None, dummy_config_gan['model']['gan']['discriminator']['conditional_dim'])]
    assert discriminator.output_shape == (None, 1)

    print("\nRCGAN model build tests completed successfully.")