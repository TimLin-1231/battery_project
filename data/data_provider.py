# battery_project/data/data_provider.py

import tensorflow as tf
import numpy as np
import logging
import os
from typing import Dict, Tuple, Optional, List, Callable

# Assuming FormatParser is defined correctly elsewhere or imported
# (FormatParser itself relies on config, so should adapt automatically)
from .preprocessing import FormatParser # Or adjust import based on your structure

logger = logging.getLogger(__name__)

# --- Data Augmentor ---
class DataAugmentor:
    """
    Applies various data augmentation techniques to time series data.
    Adapted to handle pack-level features and verify GAN output.
    """

    def __init__(self, config: Dict):
        # Overall data config needed for feature list comparison
        self.data_config = config
        # Augmentation specific config
        self.aug_config = config.get('augmentation', {})
        self.noise_level = self.aug_config.get('noise_level', 0.01)
        self.time_warp_scale = self.aug_config.get('time_warp_scale', 0.1)

        # GAN specific config
        self.gan_config = self.aug_config.get('gan', {})
        self.gan_enabled = self.gan_config.get('enabled', False)
        self.gan_probability = self.gan_config.get('probability', 0.3)
        self.generator_path = self.gan_config.get('generator_path', None)
        self.gan_conditional_dim = self.gan_config.get('conditional_dim', 1)
        # *** CRITICAL: Define the features GAN generates in config ***
        self.gan_output_features = self.gan_config.get('output_features', [])
        self.num_gan_output_features = len(self.gan_output_features)

        self.generator = None

        if self.gan_enabled:
            if not self.gan_output_features:
                 logger.error("GAN augmentation enabled but 'output_features' list is missing in gan config. Disabling GAN.")
                 self.gan_enabled = False
            elif self.generator_path and os.path.exists(self.generator_path):
                try:
                    self.generator = tf.keras.models.load_model(self.generator_path)
                    logger.info(f"Loaded RCGAN generator model from: {self.generator_path}")
                    # Infer input/output details (optional but helpful)
                    try:
                        # Verify generator output dim matches config expectation
                        gen_output_shape = self.generator.output_shape
                        if len(gen_output_shape) >= 2 and gen_output_shape[-1] != self.num_gan_output_features:
                             logger.warning(f"Loaded GAN generator output feature dimension ({gen_output_shape[-1]}) "
                                            f"does not match configured 'output_features' count ({self.num_gan_output_features}). "
                                            "Output alignment might fail.")
                        # Infer noise dim (can be complex depending on input structure)
                        self.gan_noise_dim = self.gan_config.get('noise_dim', 100) # Use config noise dim primarily
                        logger.info(f"GAN using noise_dim={self.gan_noise_dim}, conditional_dim={self.gan_conditional_dim}. Expecting generator output features: {self.gan_output_features}")

                    except Exception as e:
                         logger.warning(f"Could not fully inspect loaded GAN model: {e}. Relying solely on config.")
                         self.gan_noise_dim = self.gan_config.get('noise_dim', 100)

                except Exception as e:
                    logger.error(f"Failed to load GAN generator from {self.generator_path}: {e}", exc_info=True)
                    self.gan_enabled = False # Disable if loading failed
            else:
                logger.warning("GAN augmentation enabled but generator_path is invalid or 'output_features' not set.")
                self.gan_enabled = False

    def add_noise(self, x: tf.Tensor) -> tf.Tensor:
        """Adds Gaussian noise to the input tensor."""
        if self.noise_level > 0:
            noise = tf.random.normal(shape=tf.shape(x), stddev=self.noise_level)
            return x + noise
        return x

    def time_warp(self, x: tf.Tensor) -> tf.Tensor:
        """Applies time warping (placeholder)."""
        if self.time_warp_scale > 0:
            # Placeholder logic - replace with proper time warping implementation if needed
            pass
        return x

    def apply_gan_augmentation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generates a synthetic sequence using the loaded RCGAN generator,
        adapted for pack-level features.

        Args:
            x (np.ndarray): The original feature sequence (pack-level). Shape (seq_len, num_features).
            y (np.ndarray): The corresponding label (e.g., pack_soh), used for conditioning.

        Returns:
            np.ndarray: The generated synthetic feature sequence.
        """
        if not self.gan_enabled or self.generator is None:
            logger.warning("Attempted GAN augmentation but generator is not available.")
            return x

        try:
            # Get expected shape based on input `x` and main data config
            seq_len, num_expected_features = x.shape
            expected_feature_names = self.data_config['features']
            if num_expected_features != len(expected_feature_names):
                 logger.error(f"Input data feature count ({num_expected_features}) doesn't match config feature list length ({len(expected_feature_names)}). Aborting GAN aug.")
                 return x

            batch_size = 1

            # --- Prepare Conditional Input ---
            condition = np.reshape(y, (batch_size, self.gan_conditional_dim)).astype(np.float32)

            # --- Prepare Noise Input ---
            noise = np.random.normal(size=(batch_size, self.gan_noise_dim)).astype(np.float32)

            # --- Combine Noise and Condition (Adjust based on generator design) ---
            # Assuming Method A: noise + condition concatenated as input
            generator_input = np.concatenate([noise, condition], axis=1)

            # --- Generate Sequence ---
            synthetic_x_raw = self.generator.predict(generator_input, verbose=0)
            synthetic_x = synthetic_x_raw[0] # Get the single generated sequence

            # --- Post-process & Alignment ---
            # 1. Check Feature Dimension Matching
            gen_features_count = synthetic_x.shape[-1]
            if gen_features_count != self.num_gan_output_features:
                 logger.warning(f"GAN model internal output dim ({gen_features_count}) doesn't match configured 'output_features' count ({self.num_gan_output_features}).")
                 # Attempt to use configured count anyway, might fail if model is wrong
                 gen_features_count = self.num_gan_output_features


            if gen_features_count != num_expected_features:
                logger.warning(f"GAN generated feature count ({gen_features_count}, based on config: {self.gan_output_features}) "
                               f"does not match expected data feature count ({num_expected_features}, based on config: {expected_feature_names}).")
                # Strategy: Try to select the first `num_expected_features` if GAN generates more? Risky.
                if gen_features_count > num_expected_features:
                     logger.warning(f"  Taking first {num_expected_features} features from GAN output.")
                     synthetic_x = synthetic_x[:, :num_expected_features]
                else:
                     # Cannot proceed if GAN generates fewer features than needed
                     logger.error("  GAN generated fewer features than expected. Returning original data.")
                     return x
            else:
                 # --- TODO: Feature Order Alignment (Optional but Recommended) ---
                 # If self.gan_output_features != expected_feature_names (order differs),
                 # we need to reorder the columns of synthetic_x here based on names.
                 # Example:
                 # if self.gan_output_features != expected_feature_names:
                 #     logger.debug("Aligning GAN output feature order...")
                 #     gan_output_df = pd.DataFrame(synthetic_x, columns=self.gan_output_features)
                 #     aligned_df = gan_output_df[expected_feature_names] # Reorder/select based on expected names
                 #     synthetic_x = aligned_df.values
                 pass # Assume order matches for now if counts match


            # 2. Check Sequence Length Matching
            if synthetic_x.shape[0] != seq_len:
                 logger.warning(f"GAN generated seq_len {synthetic_x.shape[0]} != expected {seq_len}. Adjusting.")
                 if synthetic_x.shape[0] > seq_len:
                     synthetic_x = synthetic_x[:seq_len, :]
                 else:
                     padding = np.zeros((seq_len - synthetic_x.shape[0], synthetic_x.shape[1]))
                     synthetic_x = np.vstack([synthetic_x, padding])


            logger.debug("Successfully generated synthetic sequence using GAN.")
            return synthetic_x.astype(np.float32)

        except Exception as e:
            logger.error(f"Error during GAN augmentation: {e}", exc_info=True)
            return x # Return original data in case of error


    def augment(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies configured augmentations."""
        # Apply standard augmentations first
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        x_tensor = self.add_noise(x_tensor)
        x_tensor = self.time_warp(x_tensor) # Placeholder
        x = x_tensor.numpy() # Convert back for potential GAN input

        # Apply GAN augmentation probabilistically
        if self.gan_enabled and np.random.rand() < self.gan_probability:
             logger.debug("Applying GAN augmentation...")
             x = self.apply_gan_augmentation(x, y)

        return x, y


# --- Optimized Data Loader ---
class OptimizedDataLoader:
    """
    Loads, preprocesses, and batches battery pack data using tf.data.
    Relies on config for pack-level features and targets.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.model_config = config['model']
        self.batch_size = config['training']['batch_size']
        self.seq_len = self.model_config['sequence_length']
        self.predict_len = self.model_config['prediction_length']
        # Read pack-level features and target from config
        self.features = self.data_config['features']
        self.target = self.data_config['target'] # Target is now a list
        self.num_features = len(self.features)
        self.tfrecord_dir = os.path.join(self.data_config['processed_path'], 'tfrecords')

        # Instantiate augmentor
        self.augmentor = DataAugmentor(self.data_config) # Pass data config to augmentor
        # Instantiate parser - uses features/target from config
        self.parser = FormatParser(self.features, self.target, self.seq_len, self.predict_len)

        self.prefetch_buffer = tf.data.AUTOTUNE
        self.shuffle_buffer_size = self.data_config.get('shuffle_buffer_size', 1000)
        self.num_parallel_calls = tf.data.AUTOTUNE

        # Hardware adaptation from system_config
        hw_config = config.get('hardware', {})
        self.num_parallel_calls = hw_config.get('tf_data_num_parallel_calls', self.num_parallel_calls)
        self.prefetch_buffer = hw_config.get('tf_data_prefetch_buffer', self.prefetch_buffer)


    def _parse_tfrecord(self, example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Parses a single tf.train.Example proto using FormatParser."""
        return self.parser.parse_tfrecord(example_proto)

    def _augment_data_tf_wrapper(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Wraps the numpy-based augmentation function for use in tf.data.map."""
        # Define the python function to wrap
        def _augment_py_func(x_np: np.ndarray, y_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return self.augmentor.augment(x_np, y_np)

        # Wrap the function using tf.py_function
        x_aug, y_aug = tf.py_function(
            func=_augment_py_func,
            inp=[x, y],
            Tout=[tf.float32, y.dtype] # Output types
        )

        # Set the shape information for the output tensors
        # Use num_features derived from the config
        x_output_shape = tf.TensorShape([self.seq_len, self.num_features])
        # Target shape depends on prediction length
        y_output_shape = tf.TensorShape([self.predict_len])

        x_aug.set_shape(x_output_shape)
        y_aug.set_shape(y_output_shape)

        return x_aug, y_aug


    def _get_dataset_files(self, dataset_type: str) -> List[str]:
        """Gets list of TFRecord files for train, validation, or test."""
        pattern = os.path.join(self.tfrecord_dir, f"{dataset_type}_*.tfrecord")
        files = tf.io.gfile.glob(pattern)
        if not files:
            # Allow for case where maybe only training data exists, etc.
            logger.warning(f"No TFRecord files found matching pattern: {pattern}")
            # raise FileNotFoundError(f"No TFRecord files found matching pattern: {pattern}")
        else:
             logger.info(f"Found {len(files)} files for dataset type '{dataset_type}'.")
        return files

    def load_dataset(self, dataset_type: str, shuffle: bool = True, augment: bool = False,
                     repeat: bool = False, batch: bool = True, drop_remainder: bool = False) -> Optional[tf.data.Dataset]:
        """
        Loads and prepares a tf.data.Dataset for the specified type.
        Returns None if no data files are found for the type.
        """
        logger.info(f"Loading dataset: {dataset_type} (Shuffle: {shuffle}, Augment: {augment}, Repeat: {repeat}, Batch: {batch})")
        files = self._get_dataset_files(dataset_type)
        if not files:
            logger.warning(f"No data files found for '{dataset_type}', returning None dataset.")
            return None # Return None if no files

        dataset = tf.data.Dataset.from_tensor_slices(files)
        if shuffle:
            dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)

        dataset = dataset.interleave(
            lambda filepath: tf.data.TFRecordDataset(filepath, compression_type=self.data_config.get('tfrecord_compression', 'GZIP')),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=self.num_parallel_calls,
            deterministic=False
        )

        if shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size, reshuffle_each_iteration=True)

        # Apply parsing first
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=self.num_parallel_calls)

        if repeat:
            dataset = dataset.repeat()

        if augment:
             dataset = dataset.map(self._augment_data_tf_wrapper, num_parallel_calls=self.num_parallel_calls) # Parallelism might be limited by py_func

        if batch:
             # Apply batching before prefetching is generally recommended
             dataset = dataset.batch(self.batch_size, drop_remainder=drop_remainder)

        dataset = dataset.prefetch(self.prefetch_buffer)

        logger.info(f"Dataset '{dataset_type}' loading complete.")
        return dataset


# Example Usage (within train.py or evaluate.py)
if __name__ == '__main__':
    print("Testing OptimizedDataLoader with Pack Config...")
    # Updated dummy config reflecting pack features
    dummy_config = {
        'data': {
            # Use pack-level feature names
            'features': ['pack_voltage', 'pack_current', 'pack_temperature_avg', 'relative_time_s'],
            'target': ['pack_soh'],
            'processed_path': './dummy_processed_pack_data_provider', # Adjust path
            'shuffle_buffer_size': 100,
            'tfrecord_compression': 'GZIP',
            'augmentation': {
                'noise_level': 0.01,
                'time_warp_scale': 0.0,
                'gan': {
                    'enabled': True, # Test GAN path
                    'probability': 0.5,
                    'generator_path': './dummy_gan_generator_pack.keras', # Provide path
                    'conditional_dim': 1,
                    'noise_dim': 100,
                    # *** CRITICAL: List features GAN actually generates ***
                    'output_features': ['pack_voltage', 'pack_current', 'pack_temperature_avg', 'relative_time_s'] # Assume GAN generates these
                }
            }
        },
        'model': {
            'sequence_length': 50,
            'prediction_length': 1
        },
        'training': {
            'batch_size': 32
        },
        'hardware': {}
    }

    # --- Create dummy TFRecord files with pack data ---
    tfrecord_dir = os.path.join(dummy_config['data']['processed_path'], 'tfrecords')
    os.makedirs(tfrecord_dir, exist_ok=True)
    dummy_train_file = os.path.join(tfrecord_dir, 'train_dummy_pack_0.tfrecord')

    seq_len = dummy_config['model']['sequence_length']
    pred_len = dummy_config['model']['prediction_length']
    # num_features now derived from config['data']['features']
    num_features = len(dummy_config['data']['features'])
    num_records = 100

    # --- Create a dummy GAN generator matching the pack feature output ---
    gan_path = dummy_config['data']['augmentation']['gan']['generator_path']
    gan_enabled = dummy_config['data']['augmentation']['gan']['enabled']
    if gan_enabled and not os.path.exists(gan_path):
        print(f"Creating dummy GAN generator at {gan_path} for testing...")
        noise_dim = dummy_config['data']['augmentation']['gan']['noise_dim']
        cond_dim = dummy_config['data']['augmentation']['gan']['conditional_dim']
        gan_output_dim = len(dummy_config['data']['augmentation']['gan']['output_features']) # Use configured output dim

        generator_input = tf.keras.Input(shape=(noise_dim + cond_dim,))
        x = tf.keras.layers.Dense(128, activation='relu')(generator_input)
        x = tf.keras.layers.RepeatVector(seq_len)(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        # Ensure output dimension matches configured gan_output_dim
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(gan_output_dim))(x)
        dummy_generator = tf.keras.Model(inputs=generator_input, outputs=output)
        dummy_generator.save(gan_path)
        print(f"Dummy generator saved (Output dim: {gan_output_dim}).")


    print(f"Creating dummy TFRecord file: {dummy_train_file}")
    with tf.io.TFRecordWriter(dummy_train_file, options=dummy_config['data']['tfrecord_compression']) as writer:
        for i in range(num_records):
            # Create dummy sequence data (pack features)
            sequence = np.random.rand(seq_len, num_features).astype(np.float32)
            # Create dummy target data (pack soh)
            target = np.random.rand(pred_len).astype(np.float32)

            feature_dict = {
                'sequence': tf.train.Feature(float_list=tf.train.FloatList(value=sequence.flatten())),
                'target': tf.train.Feature(float_list=tf.train.FloatList(value=target.flatten()))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())
    print("Dummy TFRecord file created.")

    # --- Test DataLoader ---
    try:
        data_loader = OptimizedDataLoader(dummy_config)
        train_dataset = data_loader.load_dataset('train', shuffle=True, augment=True, batch=True) # Enable augmentation

        if train_dataset:
             print("\nDataset spec:", train_dataset.element_spec)
             # Take one batch and print shapes
             for x_batch, y_batch in train_dataset.take(1):
                 print("\nSample batch retrieved:")
                 print("x_batch shape:", x_batch.shape)
                 print("y_batch shape:", y_batch.shape)
                 # Verify batch shape
                 assert x_batch.shape == (dummy_config['training']['batch_size'], seq_len, num_features)
                 assert y_batch.shape == (dummy_config['training']['batch_size'], pred_len)

             print("\nOptimizedDataLoader test successful (check logs for augmentation details).")
        else:
             print("\nOptimizedDataLoader test failed: No dataset loaded.")


    except FileNotFoundError as e:
        print(f"\nError during testing: {e}")
        print("Please ensure dummy data paths and files are correctly set up.")
    except Exception as e:
         print(f"\nAn unexpected error occurred during testing: {e}")
    # finally: # Keep dummy files for inspection
    #     import shutil
    #     path_to_remove = dummy_config['data']['processed_path']
    #     if os.path.exists(path_to_remove):
    #          print(f"Cleaning up dummy directory: {path_to_remove}")
    #          shutil.rmtree(path_to_remove)