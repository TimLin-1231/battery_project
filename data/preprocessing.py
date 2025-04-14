# battery_project/data/preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit, LeaveOneGroupOut, GroupKFold
import tensorflow as tf
import logging
from typing import List, Dict, Tuple, Optional, Any
import joblib # For saving scaler

# Import necessary components from the project
from .cleaning import DataCleaningPipeline
from .feature_engineering import extract_health_features # Will be adapted for pack data next
from ..utils.filesystem import ensure_dir, save_dataframe # Assuming filesystem utils exist

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Orchestrates the data preprocessing pipeline:
    Load -> Clean -> Feature Engineering -> Sequence Generation -> Split -> Scale -> Save TFRecords.
    Adapted for processing battery pack data.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.model_config = config['model']
        self.train_config = config['training']
        self.cleaner = DataCleaningPipeline(config)

        # Paths
        self.raw_path = self.data_config['raw_path']
        self.interim_path = self.data_config['interim_path']
        self.processed_path = self.data_config['processed_path']
        self.results_path = self.data_config['results_path']
        self.tfrecord_dir = os.path.join(self.processed_path, 'tfrecords')
        ensure_dir(self.interim_path)
        ensure_dir(self.processed_path)
        ensure_dir(self.tfrecord_dir)
        ensure_dir(self.results_path)

        # Parameters
        self.seq_len = self.model_config['sequence_length']
        self.pred_len = self.model_config['prediction_length']
        self.feature_cols = self.data_config['features'] # Expecting pack-level features from config
        self.target_col = self.data_config['target'][0] # Assuming single target for now (e.g., 'pack_soh')
        self.cycle_col = self.data_config.get('cycle_column', 'cycle_index')
        self.group_col = self.train_config.get('group_column', None) # Column for group-based splitting (e.g., 'battery_pack_id')

        self.scaler = None
        self.scaler_path = os.path.join(self.processed_path, 'scaler.joblib')

    def process_single_file(self, filepath: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Processes a single raw data file."""
        filename = os.path.basename(filepath)
        logger.info(f"Processing file: {filename}...")
        try:
            # 1. Load Data
            # Add encoding handling based on typical battery data sources
            try:
                 raw_df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                 logger.warning(f"UTF-8 decoding failed for {filename}, trying ISO-8859-1...")
                 raw_df = pd.read_csv(filepath, encoding='iso-8859-1')
            logger.info(f"Loaded {filename} with shape {raw_df.shape}")

            # 2. Clean Data
            cleaned_df = self.cleaner.run(raw_df.copy(), filename) # Use copy to avoid modifying original

            if cleaned_df.empty:
                logger.warning(f"DataFrame became empty after cleaning {filename}. Skipping.")
                return None, None

            # Save cleaned data (optional)
            cleaned_filename = filename.replace('.csv', '_cleaned.parquet')
            save_dataframe(cleaned_df, os.path.join(self.interim_path, cleaned_filename))
            logger.info(f"Saved cleaned data for {filename} to {cleaned_filename}")

            # 3. Feature Engineering (Applied per cycle)
            # Assumes extract_health_features works on cleaned_df and returns cycle-level features
            # NOTE: feature_engineering.py needs adaptation for pack data
            logger.info(f"Starting feature engineering for {filename}...")
            if self.cycle_col not in cleaned_df.columns:
                 logger.warning(f"Cycle column '{self.cycle_col}' not found. Cannot perform cycle-based feature engineering.")
                 feature_df = pd.DataFrame() # Empty features
            else:
                 all_cycle_features = []
                 prev_cycle_df = None
                 # Group by cycle and apply feature extraction
                 for cycle_num, cycle_df in cleaned_df.groupby(self.cycle_col):
                     logger.debug(f"  Extracting features for cycle {cycle_num}...")
                     # Call the (soon to be adapted) feature extraction function
                     # Pass pack-level cycle_df
                     cycle_features = extract_health_features(
                         cycle_data=cycle_df,
                         cycle_number=int(cycle_num),
                         prev_cycle_data=prev_cycle_df,
                         config=self.config # Pass config if needed
                     )
                     # Add cycle number if not already present
                     if 'cycle_number' not in cycle_features:
                          cycle_features['cycle_number'] = int(cycle_num)

                     all_cycle_features.append(cycle_features)
                     prev_cycle_df = cycle_df.copy() # Store for next iteration's CTV calc

                 if all_cycle_features:
                     feature_df = pd.DataFrame(all_cycle_features)
                     # Merge cycle-level features back to time-series data?
                     # Option 1: Keep separate (as currently implied)
                     # Option 2: Merge feature_df back onto cleaned_df using cycle_col
                     # If merging, ensure feature_cols in config includes these new features
                     # Example merge:
                     # cleaned_df = pd.merge(cleaned_df, feature_df, on=self.cycle_col, how='left')
                     logger.info(f"Generated {feature_df.shape[0]} sets of cycle features with {feature_df.shape[1]} features each.")
                 else:
                     logger.warning(f"No cycle features generated for {filename}.")
                     feature_df = pd.DataFrame()

            # Save cycle-level features (optional)
            if not feature_df.empty:
                feature_filename = filename.replace('.csv', '_cycle_features.parquet')
                save_dataframe(feature_df, os.path.join(self.processed_path, feature_filename))
                logger.info(f"Saved cycle features for {filename} to {feature_filename}")

            # Return the cleaned time-series data and cycle features
            # The main pipeline will decide how to combine/use them later
            return cleaned_df, feature_df

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return None, None
        except Exception as e:
            logger.error(f"Failed to process file {filepath}: {e}", exc_info=True)
            return None, None


    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates sequences from the time-series data.

        Args:
            df (pd.DataFrame): DataFrame with time-series data (cleaned, potentially with merged features).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Numpy arrays for sequences (X), targets (y),
                                                     and corresponding cycle numbers for each sequence.
        """
        logger.info(f"Creating sequences with sequence length {self.seq_len} and prediction length {self.pred_len}...")
        # Ensure required columns exist
        if not all(col in df.columns for col in self.feature_cols):
             missing_cols = list(set(self.feature_cols) - set(df.columns))
             logger.error(f"Missing required feature columns for sequencing: {missing_cols}. Available: {df.columns.tolist()}")
             raise ValueError(f"Missing required feature columns: {missing_cols}")
        if self.target_col not in df.columns:
             logger.error(f"Target column '{self.target_col}' not found for sequencing.")
             raise ValueError(f"Target column '{self.target_col}' not found.")

        data_values = df[self.feature_cols].values
        target_values = df[self.target_col].values
        cycle_values = df[self.cycle_col].values if self.cycle_col in df.columns else np.zeros(len(df)) # Use cycle info

        n_samples = len(data_values)
        sequences = []
        targets = []
        seq_cycles = []

        # Use stride trick for potentially faster sequence generation (requires numpy >= 1.20)
        # try:
        #     from numpy.lib.stride_tricks import sliding_window_view
        #     if n_samples >= self.seq_len + self.pred_len -1 :
        #         seq_indices = np.arange(n_samples - self.seq_len - self.pred_len + 1)
        #         X_windows = sliding_window_view(data_values, window_shape=(self.seq_len, data_values.shape[1]))[seq_indices]
        #         y_indices = seq_indices + self.seq_len + self.pred_len - 1 # Index of the last point of the target window
        #         Y = target_values[y_indices]
        #         C = cycle_values[y_indices] # Cycle number corresponding to the target
        #         logger.info(f"Generated {len(X_windows)} sequences using sliding_window_view.")
        #         return X_windows.reshape(len(X_windows), self.seq_len, -1), Y.reshape(-1, self.pred_len), C # Ensure correct reshape for pred_len > 1

        # except ImportError:
            # Fallback to standard loop if sliding_window_view is not available
        logger.debug("Using loop-based sequence generation.")
        for i in range(n_samples - self.seq_len - self.pred_len + 1):
            seq_end = i + self.seq_len
            target_idx = seq_end + self.pred_len - 1 # Index for the target value

            sequences.append(data_values[i:seq_end])
            targets.append(target_values[target_idx]) # Get target value after the sequence
            seq_cycles.append(cycle_values[target_idx]) # Store cycle number corresponding to target

        if not sequences:
            logger.warning("No sequences were generated. Check data length and sequence parameters.")
            return np.array([]), np.array([]), np.array([])

        X = np.array(sequences).astype(np.float32)
        y = np.array(targets).astype(np.float32).reshape(-1, self.pred_len) # Reshape for consistency
        cycles = np.array(seq_cycles)

        logger.info(f"Generated {len(X)} sequences. X shape: {X.shape}, y shape: {y.shape}, cycles shape: {cycles.shape}")
        return X, y, cycles


    def split_data(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> \
            List[Tuple[np.ndarray, np.ndarray]]:
        """
        Splits data into train, validation, and test sets based on config strategy.

        Args:
            X (np.ndarray): Feature sequences.
            y (np.ndarray): Target values.
            groups (Optional[np.ndarray]): Group identifiers for each sequence (e.g., battery pack ID).

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List containing (X_train, y_train),
                                                 (X_val, y_val), (X_test, y_test).
        """
        strategy = self.train_config.get('split_strategy', 'random')
        val_split = self.train_config.get('validation_split', 0.15)
        test_split = self.train_config.get('test_split', 0.15)
        random_state = self.config.get('seed', 42)

        logger.info(f"Splitting data using strategy: {strategy} (Val: {val_split}, Test: {test_split})")

        if strategy == 'random':
            if val_split + test_split >= 1.0:
                 raise ValueError("Sum of validation_split and test_split must be less than 1.0 for random strategy.")
            # First split into train+val and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_state, shuffle=True)
            # Then split train+val into train and val
            relative_val_split = val_split / (1.0 - test_split)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=relative_val_split, random_state=random_state, shuffle=True)

        elif strategy in ['group_k_fold', 'leave_one_out', 'group_shuffle_split'] and groups is not None:
            if self.group_col is None:
                 raise ValueError("group_column must be specified in config for group-based splitting.")
            logger.info(f"Using group column: {self.group_col}")
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            logger.info(f"Found {n_groups} unique groups.")

            if n_groups < 2:
                raise ValueError("Need at least 2 groups for group-based splitting.")

            indices = np.arange(len(X))

            if strategy == 'group_shuffle_split':
                # Split into train+val and test based on groups
                gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
                train_val_idx, test_idx = next(gss_test.split(X, y, groups))
                X_train_val, X_test = X[train_val_idx], X[test_idx]
                y_train_val, y_test = y[train_val_idx], y[test_idx]
                groups_train_val = groups[train_val_idx]

                # Split train+val into train and val based on remaining groups
                relative_val_split = val_split / (1.0 - test_split)
                gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_split, random_state=random_state)
                train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            elif strategy == 'group_k_fold':
                 n_folds = self.train_config.get('num_folds', 5)
                 if n_groups < n_folds:
                      raise ValueError(f"Number of groups ({n_groups}) is less than n_folds ({n_folds}) for GroupKFold.")
                 gkf = GroupKFold(n_splits=n_folds)
                 # Use one fold for test, one for val, rest for train
                 fold_indices = list(gkf.split(X, y, groups))
                 test_idx = fold_indices[0][1] # Use first fold's test set as overall test
                 val_idx = fold_indices[1][1]  # Use second fold's test set as overall val
                 # Combine remaining folds for training
                 train_idx_list = [fold_indices[i][0] for i in range(n_folds) if i > 1] \
                                + [fold_indices[0][0], fold_indices[1][0]] # Add train indices from fold 0 & 1
                 # Need intersection logic if indices overlap across folds' train sets
                 # Simplification: Assume train indices from fold 2 onwards cover most?
                 # A safer approach: Concatenate all indices and use test/val indices to exclude
                 all_indices = np.arange(len(X))
                 train_idx = np.setdiff1d(all_indices, np.concatenate([test_idx, val_idx]))

                 X_train, y_train = X[train_idx], y[train_idx]
                 X_val, y_val = X[val_idx], y[val_idx]
                 X_test, y_test = X[test_idx], y[test_idx]

            elif strategy == 'leave_one_out':
                 if n_groups < 3: raise ValueError("LeaveOneGroupOut requires at least 3 groups for train/val/test split.")
                 logo = LeaveOneGroupOut()
                 splits = list(logo.split(X, y, groups))
                 # Use first group for test, second for validation, rest for train
                 test_indices = splits[0][1]
                 val_indices = splits[1][1]
                 train_indices = np.concatenate([splits[i][0] for i in range(n_groups) if i not in [0, 1]])
                 # Verify indices are unique and cover all data?

                 X_train, y_train = X[train_indices], y[train_indices]
                 X_val, y_val = X[val_indices], y[val_indices]
                 X_test, y_test = X[test_indices], y[test_indices]

            else:
                 raise ValueError(f"Unsupported group split strategy: {strategy}")

        else: # Random or group strategy without groups provided
            logger.warning(f"Group column '{self.group_col}' not found or strategy is random. Using random split.")
            # Fallback to random split
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_state, shuffle=True)
            relative_val_split = val_split / (1.0 - test_split)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=relative_val_split, random_state=random_state, shuffle=True)

        logger.info(f"Split complete: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
        return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]

    def fit_scaler(self, X_train: np.ndarray):
        """Fits the StandardScaler on the training data."""
        if not self.feature_cols:
             logger.warning("No feature columns defined. Skipping scaler fitting.")
             return
        logger.info(f"Fitting StandardScaler on {len(X_train)} training sequences for features: {self.feature_cols}")
        # Scaler expects 2D array: (n_samples * seq_len, n_features)
        n_features = X_train.shape[2]
        X_train_reshaped = X_train.reshape(-1, n_features)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_reshaped)
        logger.info(f"Scaler fitted. Mean: {self.scaler.mean_}, Scale: {self.scaler.scale_}")
        # Save the scaler
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")

    def apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Applies the fitted StandardScaler to data."""
        if self.scaler is None:
            logger.warning("Scaler has not been fitted. Returning original data.")
            return X
        if X.ndim != 3:
             raise ValueError(f"Input array X must be 3D (sequences, timesteps, features), but got shape {X.shape}")

        n_sequences, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled_reshaped = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape(n_sequences, seq_len, n_features)
        logger.debug(f"Applied scaler to data with shape {X.shape}")
        return X_scaled

    def _write_tfrecord(self, X: np.ndarray, y: np.ndarray, filename: str):
        """Writes sequences and targets to a TFRecord file."""
        filepath = os.path.join(self.tfrecord_dir, filename)
        compression = self.data_config.get('tfrecord_compression', 'GZIP')
        logger.info(f"Writing {len(X)} sequences to TFRecord: {filepath} (Compression: {compression})")

        with tf.io.TFRecordWriter(filepath, options=compression) as writer:
            for i in range(len(X)):
                sequence = X[i].flatten() # Flatten the sequence
                target = y[i].flatten()   # Flatten the target (handles pred_len > 1)

                feature_dict = {
                    'sequence': tf.train.Feature(float_list=tf.train.FloatList(value=sequence)),
                    'target': tf.train.Feature(float_list=tf.train.FloatList(value=target))
                    # Add cycle number or other metadata if needed
                    # 'cycle': tf.train.Feature(int64_list=tf.train.Int64List(value=[cycles[i]]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                writer.write(example.SerializeToString())
        logger.debug(f"Finished writing {filepath}")


    def run_pipeline(self):
        """Executes the full preprocessing pipeline."""
        logger.info("Starting full preprocessing pipeline...")
        all_files = [os.path.join(self.raw_path, f) for f in os.listdir(self.raw_path) if f.endswith('.csv')] # Example: find CSVs
        if not all_files:
            logger.error(f"No raw data files (.csv) found in {self.raw_path}. Aborting.")
            return

        all_cleaned_dfs = []
        all_feature_dfs = [] # To store cycle-level features if generated

        # --- 1. Process each file (Load, Clean, Feature Eng) ---
        for filepath in all_files:
            cleaned_df, feature_df = self.process_single_file(filepath)
            if cleaned_df is not None:
                # Add identifier for the source file/battery pack
                pack_id = os.path.splitext(os.path.basename(filepath))[0] # Use filename as ID
                cleaned_df[self.group_col or 'battery_pack_id'] = pack_id # Add group ID
                all_cleaned_dfs.append(cleaned_df)
                if feature_df is not None and not feature_df.empty:
                     feature_df[self.group_col or 'battery_pack_id'] = pack_id
                     all_feature_dfs.append(feature_df)

        if not all_cleaned_dfs:
            logger.error("No dataframes were successfully processed. Aborting.")
            return

        # --- 2. Combine Data ---
        logger.info(f"Combining data from {len(all_cleaned_dfs)} processed files...")
        combined_df = pd.concat(all_cleaned_dfs, ignore_index=True)
        logger.info(f"Combined time-series data shape: {combined_df.shape}")
        # --- Optional: Combine and merge cycle features ---
        if all_feature_dfs:
             combined_feature_df = pd.concat(all_feature_dfs, ignore_index=True)
             logger.info(f"Combined cycle features shape: {combined_feature_df.shape}")
             # Decide whether/how to merge features into combined_df here
             # Example: Merge based on cycle and group ID
             # if self.cycle_col in combined_df.columns and self.cycle_col in combined_feature_df.columns \
             #    and (self.group_col or 'battery_pack_id') in combined_df.columns \
             #    and (self.group_col or 'battery_pack_id') in combined_feature_df.columns:
             #     combined_df = pd.merge(combined_df, combined_feature_df,
             #                           on=[self.cycle_col, self.group_col or 'battery_pack_id'],
             #                           how='left')
             #     logger.info("Merged cycle features into main dataframe.")
             #     # Update self.feature_cols if new features were merged and should be used by model
             #     # self.feature_cols.extend([col for col in combined_feature_df.columns if col not in combined_df.columns])

        # --- 3. Create Sequences ---
        X, y, cycles_or_groups = self.create_sequences(combined_df)
        if X.size == 0:
            logger.error("Sequence creation failed. Aborting.")
            return

        # Use appropriate groups for splitting
        groups = combined_df.loc[cycles_or_groups.index, self.group_col or 'battery_pack_id'].values \
                if self.train_config.get('split_strategy', 'random') != 'random' \
                else None


        # --- 4. Split Data ---
        splits = self.split_data(X, y, groups=groups)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

        # --- 5. Fit and Apply Scaler ---
        self.fit_scaler(X_train)
        X_train_scaled = self.apply_scaler(X_train)
        X_val_scaled = self.apply_scaler(X_val)
        X_test_scaled = self.apply_scaler(X_test)

        # --- 6. Save TFRecords ---
        self._write_tfrecord(X_train_scaled, y_train, 'train_data.tfrecord')
        self._write_tfrecord(X_val_scaled, y_val, 'validation_data.tfrecord')
        self._write_tfrecord(X_test_scaled, y_test, 'test_data.tfrecord')

        logger.info("Preprocessing pipeline completed successfully!")


class FormatParser:
    """Parses TFRecord examples based on configuration."""

    def __init__(self, feature_cols: List[str], target_col: List[str], seq_len: int, pred_len: int):
        self.feature_cols = feature_cols
        self.target_col = target_col # Assumes target_col is a list from config
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = len(feature_cols)

        # Define the feature description for parsing TFRecords
        self.feature_spec = {
            'sequence': tf.io.FixedLenFeature([self.seq_len * self.num_features], tf.float32),
            'target': tf.io.FixedLenFeature([self.pred_len], tf.float32) # Assumes target is flattened to pred_len
            # Add other features saved in TFRecord (e.g., 'cycle') if needed
            # 'cycle': tf.io.FixedLenFeature([], tf.int64)
        }
        logger.debug(f"FormatParser initialized. Feature spec: {self.feature_spec}")


    def parse_tfrecord(self, example_proto: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Parses a single tf.train.Example proto and reshapes the sequence."""
        parsed_features = tf.io.parse_single_example(example_proto, self.feature_spec)
        sequence = parsed_features['sequence']
        target = parsed_features['target']

        # Reshape sequence back to (seq_len, num_features)
        sequence_reshaped = tf.reshape(sequence, (self.seq_len, self.num_features))

        # Ensure target has the correct shape (pred_len,) - it's already [pred_len] from FixedLenFeature
        # target_reshaped = tf.reshape(target, (self.pred_len,)) # Not needed if FixedLenFeature is correct

        # tf.print("Parsed sequence shape:", tf.shape(sequence_reshaped)) # Debug print
        # tf.print("Parsed target shape:", tf.shape(target)) # Debug print
        return sequence_reshaped, target

# --- Example Usage ---
if __name__ == '__main__':
    print("Testing DataProcessor pipeline...")
     # Create dummy config reflecting pack structure
    dummy_config = {
        'project_name': 'PackPreprocessTest',
        'seed': 42,
        'pack_config': { 'num_cells_series': 13, 'nominal_capacity_ah': 6.2},
        'physics_config': {},
        'data': {
            'raw_path': './dummy_raw_pack_data', # Create dummy dir
            'interim_path': './dummy_interim_pack_data',
            'processed_path': './dummy_processed_pack_data',
            'results_path': './dummy_results_pack_data',
            'log_path': './dummy_logs_pack_data',
            'model_save_path': './dummy_models_pack_data',
            'features': ['pack_voltage', 'pack_current', 'pack_temperature_avg', 'relative_time_s'], # Pack features
            'target': ['pack_soh'],
            'time_column': 'timestamp',
            'cycle_column': 'cycle_index',
            'step_column': 'step_type',
            'voltage_range': [30.0, 55.0],
            'current_range': [-20, 20],
            'temperature_range': [0, 60],
            'column_mapping': {'V': 'pack_voltage', 'I': 'pack_current', 'T': 'pack_temperature_avg', 'SoH':'pack_soh'},
             'tfrecord_compression': 'GZIP',
        },
        'model': {
            'sequence_length': 20, # Shorter seq len for faster test
            'prediction_length': 1
        },
        'training': {
             'optimizer': 'adam',
             'learning_rate': 1e-3,
             'batch_size': 16,
             'epochs': 1,
             'validation_split': 0.2,
             'test_split': 0.2,
             'split_strategy': 'group_shuffle_split', # Test group split
             'group_column': 'pack_id', # Use pack ID for splitting
        }
        # Add other sections if needed by the classes
    }

    # --- Create dummy raw files ---
    raw_dir = dummy_config['data']['raw_path']
    ensure_dir(raw_dir)
    num_files = 3
    rows_per_file = 100
    seq_len = dummy_config['model']['sequence_length']
    pred_len = dummy_config['model']['prediction_length']

    for i in range(num_files):
        pack_id = f'pack_{i+1}'
        file_path = os.path.join(raw_dir, f'{pack_id}.csv')
        df_data = {
            'timestamp': pd.date_range(start='2025-01-01', periods=rows_per_file, freq='10S'),
            'cycle_index': np.repeat(np.arange(1, rows_per_file // 20 + 1), 20)[:rows_per_file],
             'step_type': np.random.choice(['charge', 'discharge'], size=rows_per_file),
            'V': np.random.uniform(35, 54, size=rows_per_file),
            'I': np.random.uniform(-10, 10, size=rows_per_file),
            'T': np.random.uniform(15, 45, size=rows_per_file),
             'SoH': np.linspace(1.0, 0.95 - i*0.02, num=rows_per_file) # Simulate degradation per pack
        }
        dummy_df = pd.DataFrame(df_data)
        dummy_df.to_csv(file_path, index=False)
        print(f"Created dummy file: {file_path}")

    # --- Run the processor ---
    try:
        processor = DataProcessor(dummy_config)
        processor.run_pipeline()

        # --- Verify output ---
        tfrecord_dir = os.path.join(dummy_config['data']['processed_path'], 'tfrecords')
        assert os.path.exists(os.path.join(tfrecord_dir, 'train_data.tfrecord'))
        assert os.path.exists(os.path.join(tfrecord_dir, 'validation_data.tfrecord'))
        assert os.path.exists(os.path.join(tfrecord_dir, 'test_data.tfrecord'))
        assert os.path.exists(os.path.join(dummy_config['data']['processed_path'], 'scaler.joblib'))
        print("\nDataProcessor pipeline test completed successfully. Check dummy folders for output.")

        # --- Test FormatParser ---
        print("\nTesting FormatParser...")
        features = dummy_config['data']['features']
        target = dummy_config['data']['target']
        parser = FormatParser(features, target, seq_len, pred_len)

        # Create a dummy dataset from TFRecords to test parsing
        test_record_file = os.path.join(tfrecord_dir, 'test_data.tfrecord')
        if os.path.exists(test_record_file):
             dataset = tf.data.TFRecordDataset(test_record_file, compression_type='GZIP')
             parsed_dataset = dataset.map(parser.parse_tfrecord)

             for seq, targ in parsed_dataset.take(1):
                 print("Parsed sequence shape:", seq.shape)
                 print("Parsed target shape:", targ.shape)
                 assert seq.shape == (seq_len, len(features)), "Parsed sequence shape mismatch"
                 assert targ.shape == (pred_len,), "Parsed target shape mismatch"
             print("FormatParser test successful.")
        else:
             print("Could not find test TFRecord file to test parser.")


    except Exception as e:
        print(f"\nError during DataProcessor test: {e}", exc_info=True)
    finally:
        # Clean up dummy directories
        import shutil
        for path_key in ['raw_path', 'interim_path', 'processed_path', 'results_path', 'log_path', 'model_save_path']:
             path_to_remove = dummy_config['data'].get(path_key)
             if path_to_remove and os.path.exists(path_to_remove):
                 print(f"Cleaning up dummy directory: {path_to_remove}")
                 shutil.rmtree(path_to_remove)