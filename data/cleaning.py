# battery_project/data/cleaning.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Default Column Mapping (Update based on your actual raw data column names) ---
# Maps common raw data column names to standardized internal names (now pack-focused)
COLUMN_MAPPING = {
    # Potential Raw Names : Standard Name
    'Voltage(V)': 'pack_voltage',
    'Pack Voltage': 'pack_voltage',
    'Total Voltage': 'pack_voltage',
    'Current(A)': 'pack_current',
    'Pack Current': 'pack_current',
    'Total Current': 'pack_current',
    'Temperature(C)': 'pack_temperature_avg',
    'Pack Temperature': 'pack_temperature_avg',
    'Avg Pack Temp': 'pack_temperature_avg',
    'Timestamp': 'timestamp',
    'Time': 'timestamp',
    'Record Time': 'timestamp',
    'Cycle Index': 'cycle_index',
    'Cycle': 'cycle_index',
    'Charge Capacity(Ah)': 'charge_capacity', # May represent pack capacity
    'Discharge Capacity(Ah)': 'discharge_capacity', # May represent pack capacity
    'Step Type': 'step_type', # e.g., 'charge', 'discharge', 'rest'
    'Step': 'step_type',
    'Relative Time(s)': 'step_time_s', # Time within a step
    'Test Time(s)': 'total_time_s',   # Total experiment time
    # Add other mappings as needed
}


class DataCleaningPipeline:
    """
    A pipeline for cleaning raw battery pack cycling data.
    Focuses on handling pack-level measurements.
    """

    def __init__(self, config: Dict):
        """
        Initializes the cleaning pipeline with configuration.

        Args:
            config (Dict): The main configuration dictionary, expected to contain
                           a 'data' key with cleaning parameters.
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.column_mapping = self.data_config.get('column_mapping', COLUMN_MAPPING)

        # Get expected feature columns (pack level) + target + identifiers
        self.expected_columns = set(self.data_config.get('features', []) + \
                                   self.data_config.get('target', []) + \
                                   [self.data_config.get('time_column', 'timestamp'),
                                    self.data_config.get('cycle_column', 'cycle_index'),
                                    self.data_config.get('step_column', 'step_type')])
        # Add common intermediate columns generated during cleaning/processing
        self.expected_columns.update(['charge_capacity', 'discharge_capacity', 'step_time_s', 'total_time_s', 'relative_time_s'])
        # Remove None if any config values were None
        self.expected_columns.discard(None)

    def run(self, df: pd.DataFrame, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Executes the full data cleaning pipeline.

        Args:
            df (pd.DataFrame): The raw input DataFrame.
            filename (Optional[str]): Original filename for logging purposes.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        log_prefix = f"[{filename}] " if filename else ""
        logger.info(f"{log_prefix}Starting data cleaning process...")
        original_rows = len(df)

        # --- 1. Unify Column Names ---
        df = self.unify_column_names(df)
        logger.debug(f"{log_prefix}Columns after unification: {df.columns.tolist()}")

        # --- 2. Handle Timestamps ---
        time_col = self._get_config_col_name('time_column', 'timestamp')
        df = self.handle_timestamps(df, time_col)
        logger.debug(f"{log_prefix}Timestamp handling complete.")

        # --- 3. Sort Data ---
        # Sort primarily by time, potentially by cycle/step if needed
        cycle_col = self._get_config_col_name('cycle_column', 'cycle_index')
        if time_col in df.columns:
            sort_keys = [time_col]
            if cycle_col in df.columns:
                 sort_keys.insert(0, cycle_col) # Sort by cycle first if available
            logger.debug(f"{log_prefix}Sorting data by {sort_keys}...")
            df = df.sort_values(by=sort_keys).reset_index(drop=True)
        else:
            logger.warning(f"{log_prefix}Cannot sort data: Time column '{time_col}' not found.")

        # --- 4. Calculate Relative/Total Time (if not present) ---
        df = self.calculate_time_columns(df, time_col)

        # --- 5. Handle Missing Values ---
        # Identify numeric columns based on config features (or infer)
        numeric_cols = self._get_numeric_columns(df)
        logger.debug(f"{log_prefix}Identified numeric columns for imputation: {numeric_cols}")
        df = self.handle_missing_values(df, numeric_cols=numeric_cols)
        logger.debug(f"{log_prefix}Missing value handling complete.")

        # --- 6. Correct Data Types ---
        df = self.correct_data_types(df, numeric_cols=numeric_cols)
        logger.debug(f"{log_prefix}Data type correction complete.")

        # --- 7. Mark Charge/Discharge Steps ---
        current_col = self._get_config_col_name('pack_current_col', 'pack_current') # Use pack current
        step_col = self._get_config_col_name('step_column', 'step_type')
        df = self.mark_charge_discharge(df, current_col=current_col, step_col=step_col)
        logger.debug(f"{log_prefix}Charge/discharge step marking complete.")

        # --- 8. Apply Physical Constraints ---
        # Uses ranges defined for the pack in config
        df = self.apply_physical_constraints(df)
        rows_after_constraints = len(df)
        if rows_after_constraints < original_rows:
            logger.info(f"{log_prefix}Removed {original_rows - rows_after_constraints} rows due to physical constraint violations.")

        # --- 9. Handle Outliers ---
        # Applies to specified pack-level numeric columns
        df = self.handle_outliers(df, numeric_cols=[col for col in numeric_cols if col in df.columns])
        logger.debug(f"{log_prefix}Outlier handling complete.")

        # --- 10. Final Checks & Column Selection ---
        df = self.final_checks(df)
        logger.info(f"{log_prefix}Data cleaning finished. Final shape: {df.shape}")

        return df

    def _get_config_col_name(self, config_key: str, default: str) -> str:
        """Gets the column name from config, providing a default."""
        return self.data_config.get(config_key, default)

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Identifies numeric columns based on config features or DataFrame types."""
        # Prioritize features listed in config
        numeric_cols = [col for col in self.data_config.get('features', []) if col in df.columns]
        # Add target if numeric
        target_cols = self.data_config.get('target', [])
        numeric_cols.extend([col for col in target_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])])
        # Add common numeric cols if not already included
        common_numeric = ['pack_voltage', 'pack_current', 'pack_temperature_avg',
                          'charge_capacity', 'discharge_capacity',
                          'cycle_index', 'step_time_s', 'total_time_s', 'relative_time_s']
        numeric_cols.extend([col for col in common_numeric if col in df.columns and col not in numeric_cols])

        # Fallback: If no features specified, infer from dtype
        if not numeric_cols:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Ensure uniqueness
        return list(set(numeric_cols))


    def unify_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames columns based on the provided mapping."""
        logger.debug("Unifying column names...")
        # Invert mapping: {StandardName: [RawName1, RawName2]}
        inverted_mapping = {}
        for raw_name, std_name in self.column_mapping.items():
            if std_name not in inverted_mapping:
                inverted_mapping[std_name] = []
            inverted_mapping[std_name].append(raw_name)

        rename_dict = {}
        used_raw_columns = set()

        for std_name, raw_names in inverted_mapping.items():
            found = False
            # Iterate through possible raw names for the current standard name
            for raw_name in raw_names:
                if raw_name in df.columns and raw_name not in used_raw_columns:
                    rename_dict[raw_name] = std_name
                    used_raw_columns.add(raw_name)
                    logger.debug(f"  Mapping '{raw_name}' to '{std_name}'")
                    found = True
                    break # Found a mapping for this standard name
            # if not found:
            #     logger.debug(f"  Standard column '{std_name}' not found in raw data columns: {raw_names}")

        # Rename based on the found mappings
        df = df.rename(columns=rename_dict)

        # Log columns that were not renamed (potential issues)
        original_cols = set(rename_dict.keys())
        remaining_cols = set(df.columns) - set(rename_dict.values()) - (set(df.columns) - original_cols)
        if remaining_cols:
            logger.debug(f"  Columns not mapped or already standard: {list(remaining_cols)}")

        return df


    def handle_timestamps(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Converts the timestamp column to datetime objects."""
        if time_col in df.columns:
            logger.debug(f"Converting column '{time_col}' to datetime...")
            try:
                # Attempt conversion, inferring format
                original_dtype = df[time_col].dtype
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                # Check for NaT values introduced by coercion
                nat_count = df[time_col].isnull().sum()
                if nat_count > 0:
                    logger.warning(f"Column '{time_col}' contained {nat_count} values that could not be parsed as dates and were set to NaT.")
                    # Optional: Fill NaT values (e.g., forward fill, backward fill)
                    # df[time_col] = df[time_col].ffill().bfill()
            except Exception as e:
                logger.error(f"Failed to convert timestamp column '{time_col}' (dtype: {original_dtype}): {e}. Please check data format.", exc_info=True)
                # Depending on severity, either raise error or continue with original column
        else:
            logger.warning(f"Timestamp column '{time_col}' not found. Time-based operations might fail.")
        return df

    def calculate_time_columns(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Calculates relative and total time in seconds if not present."""
        if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            first_time = df[time_col].min()
            if 'total_time_s' not in df.columns:
                logger.debug("Calculating 'total_time_s'...")
                df['total_time_s'] = (df[time_col] - first_time).dt.total_seconds()

            cycle_col = self._get_config_col_name('cycle_column', 'cycle_index')
            step_col = self._get_config_col_name('step_column', 'step_type') # Need step col too

            # Calculate relative time within each cycle/step group
            group_cols = []
            if cycle_col in df.columns: group_cols.append(cycle_col)
            if step_col in df.columns: group_cols.append(step_col)

            if group_cols and 'step_time_s' not in df.columns:
                 logger.debug(f"Calculating 'step_time_s' relative to start of each group in {group_cols}...")
                 try:
                     df['step_time_s'] = df.groupby(group_cols)[time_col].transform(lambda x: (x - x.min()).dt.total_seconds())
                 except Exception as e:
                      logger.warning(f"Could not calculate step_time_s due to grouping error: {e}. Skipping.")

            if 'relative_time_s' not in df.columns and cycle_col in df.columns:
                 logger.debug(f"Calculating 'relative_time_s' relative to start of each cycle...")
                 try:
                     df['relative_time_s'] = df.groupby(cycle_col)[time_col].transform(lambda x: (x - x.min()).dt.total_seconds())
                 except Exception as e:
                     logger.warning(f"Could not calculate relative_time_s due to cycle grouping error: {e}. Skipping.")

        return df

    def handle_missing_values(self, df: pd.DataFrame, numeric_cols: List[str],
                              strategy: str = 'interpolate') -> pd.DataFrame:
        """Handles missing values in numeric columns."""
        logger.debug(f"Handling missing values using strategy: {strategy}...")
        missing_counts = df[numeric_cols].isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if not cols_with_missing.empty:
            logger.warning(f"Missing values found in columns: {cols_with_missing.to_dict()}")
            if strategy == 'interpolate':
                # Use linear interpolation, limit direction for time-series appropriateness
                logger.debug("Applying linear interpolation (method='linear')...")
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='forward', axis=0)
                # Apply backward fill for any remaining NaNs at the beginning
                df[numeric_cols] = df[numeric_cols].bfill()
            elif strategy == 'ffill':
                logger.debug("Applying forward fill followed by backward fill...")
                df[numeric_cols] = df[numeric_cols].ffill().bfill()
            elif strategy == 'mean':
                logger.debug("Filling with mean values...")
                for col in cols_with_missing.index:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    logger.debug(f"  Filled NaNs in '{col}' with mean value: {mean_val:.4f}")
            elif strategy == 'zero':
                logger.debug("Filling with zero...")
                df[numeric_cols] = df[numeric_cols].fillna(0)
            else:
                logger.warning(f"Unsupported missing value strategy: {strategy}. No filling performed.")

            # Final check
            final_missing = df[numeric_cols].isnull().sum().sum()
            if final_missing > 0:
                logger.error(f"Still {final_missing} missing values remaining after applying strategy '{strategy}'. Check data or strategy.")
        else:
            logger.debug("No missing values found in specified numeric columns.")

        return df

    def correct_data_types(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Attempts to convert columns to appropriate types."""
        logger.debug("Correcting data types...")
        for col in df.columns:
            if col in numeric_cols:
                if df[col].dtype != 'float64' and df[col].dtype != 'float32': # Allow float32 for memory efficiency
                    try:
                        # Attempt conversion to numeric, coerce errors to NaN
                        original_dtype = df[col].dtype
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if df[col].isnull().any():
                             logger.warning(f"Coerced non-numeric values to NaN in column '{col}' during type conversion from {original_dtype}.")
                             # Optionally refill NaNs introduced here
                             # df[col] = df[col].fillna(0) # Example: fill with 0
                    except Exception as e:
                        logger.error(f"Could not convert column '{col}' to numeric: {e}")
            elif col == self._get_config_col_name('time_column', 'timestamp'):
                # Already handled by handle_timestamps, ensure it's datetime
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    logger.warning(f"Timestamp column '{col}' is not datetime dtype after handling.")
            elif col == self._get_config_col_name('step_column', 'step_type'):
                # Convert step type to category for efficiency
                if df[col].dtype != 'category':
                    df[col] = df[col].astype('category')
            # Add specific type corrections for other columns if needed (e.g., cycle_index to int)
            elif col == self._get_config_col_name('cycle_column', 'cycle_index'):
                 if pd.api.types.is_numeric_dtype(df[col]):
                     # Convert to integer if possible (after handling NaNs)
                     if not df[col].isnull().any():
                         df[col] = df[col].astype(int)
                 else:
                     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int) # Coerce and fill missing cycle with -1


        logger.debug(f"Final column dtypes:\n{df.dtypes}")
        return df


    def mark_charge_discharge(self, df: pd.DataFrame, current_col: str, step_col: str) -> pd.DataFrame:
        """Adds/updates a 'step_type' column based on current direction."""
        if current_col not in df.columns:
            logger.warning(f"Current column '{current_col}' not found. Cannot reliably mark charge/discharge steps.")
            if step_col not in df.columns:
                df[step_col] = 'unknown' # Create column if it doesn't exist
            return df

        logger.debug(f"Marking charge/discharge/rest steps based on '{current_col}'...")
        threshold = self.data_config.get('charge_discharge_threshold', 0.1) # Amps
        logger.debug(f"Using current threshold: {threshold} A")

        # Define conditions
        is_charge = df[current_col] > threshold
        is_discharge = df[current_col] < -threshold
        # is_rest = (df[current_col] >= -threshold) & (df[current_col] <= threshold)

        # Assign step type
        # Initialize with 'rest' or existing values if column exists
        if step_col not in df.columns:
            df[step_col] = 'rest'

        df.loc[is_charge, step_col] = 'charge'
        df.loc[is_discharge, step_col] = 'discharge'
        # Keep existing 'rest' or other labels if not charge/discharge

        # Convert to category
        df[step_col] = df[step_col].astype('category')
        logger.debug(f"Step counts: {df[step_col].value_counts().to_dict()}")
        return df

    def apply_physical_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes data points that violate physical constraints defined in config."""
        logger.debug("Applying physical constraints...")
        original_rows = len(df)
        constraints = {
            'pack_voltage': self.data_config.get('voltage_range', None),
            'pack_current': self.data_config.get('current_range', None),
            'pack_temperature_avg': self.data_config.get('temperature_range', None),
            # Add constraints for other configured features if needed
        }
        # --- TODO: Handle Individual Cell Constraints ---
        # If individual cell data exists (e.g., 'cell_1_voltage', ... 'cell_13_voltage'),
        # apply constraints to each cell column. Example:
        # num_cells = self.config.get('pack_config', {}).get('num_cells_series', 1)
        # cell_v_range = [min_cell_v, max_cell_v] # Define appropriate single cell voltage range
        # for i in range(1, num_cells + 1):
        #     col_name = f'cell_{i}_voltage'
        #     if col_name in df.columns:
        #         mask = (df[col_name] >= cell_v_range[0]) & (df[col_name] <= cell_v_range[1])
        #         removed_count = len(df) - mask.sum()
        #         if removed_count > 0:
        #              logger.debug(f"  Removing {removed_count} rows violating constraints for {col_name} ({cell_v_range})")
        #              df = df[mask]
        # Apply similarly for individual cell temperatures if available.

        for col, value_range in constraints.items():
            if value_range and col in df.columns:
                min_val, max_val = value_range
                if min_val is not None and max_val is not None:
                    mask = (df[col] >= min_val) & (df[col] <= max_val)
                    removed_count = len(df) - mask.sum()
                    if removed_count > 0:
                        logger.debug(f"  Removing {removed_count} rows violating constraints for {col} (Range: {min_val} - {max_val})")
                        df = df[mask]
                elif min_val is not None: # Only min constraint
                     mask = df[col] >= min_val
                     # ... (apply mask)
                elif max_val is not None: # Only max constraint
                     mask = df[col] <= max_val
                     # ... (apply mask)

        rows_removed = original_rows - len(df)
        if rows_removed > 0:
             logger.info(f"Removed {rows_removed} rows due to physical constraint violations on pack-level data.")
        else:
             logger.debug("No rows removed due to physical constraints.")
        return df

    def handle_outliers(self, df: pd.DataFrame, numeric_cols: List[str],
                          method: str = 'std_dev', threshold: float = 3.0) -> pd.DataFrame:
        """Handles outliers in numeric columns using specified method."""
        logger.debug(f"Handling outliers using method: {method} with threshold: {threshold}...")
        original_rows = len(df)
        outlier_indices = set()

        # --- TODO: Handle Individual Cell Outliers ---
        # Similar to constraints, apply outlier detection per cell if data is available.
        # num_cells = self.config.get('pack_config', {}).get('num_cells_series', 1)
        # for i in range(1, num_cells + 1):
        #     for base_col in ['voltage', 'temperature']: # Check relevant cols
        #         col_name = f'cell_{i}_{base_col}'
        #         if col_name in df.columns:
        #             # Apply outlier detection logic (e.g., std_dev) to col_name
        #             # mean, std = df[col_name].mean(), df[col_name].std()
        #             # col_outliers = df[(df[col_name] < mean - threshold * std) | (df[col_name] > mean + threshold * std)].index
        #             # outlier_indices.update(col_outliers)
        #             pass # Placeholder

        if method == 'std_dev':
            for col in numeric_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 1e-9: # Avoid division by zero for constant columns
                         lower_bound = mean - threshold * std
                         upper_bound = mean + threshold * std
                         col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                         if not col_outliers.empty:
                              logger.debug(f"  Detected {len(col_outliers)} potential outliers in '{col}' (Bounds: {lower_bound:.3f} - {upper_bound:.3f})")
                              outlier_indices.update(col_outliers)
                    else:
                         logger.debug(f"  Skipping outlier detection for near-constant column '{col}'")

        elif method == 'iqr':
            for col in numeric_cols:
                 if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 1e-9:
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                        if not col_outliers.empty:
                              logger.debug(f"  Detected {len(col_outliers)} potential outliers in '{col}' using IQR (Bounds: {lower_bound:.3f} - {upper_bound:.3f})")
                              outlier_indices.update(col_outliers)
                    else:
                         logger.debug(f"  Skipping outlier detection for column '{col}' with IQR close to zero.")
        else:
            logger.warning(f"Unsupported outlier detection method: {method}. No outlier handling performed.")

        if outlier_indices:
            logger.info(f"Removing {len(outlier_indices)} rows identified as outliers.")
            df = df.drop(index=list(outlier_indices))
        else:
            logger.debug("No outliers detected or removed.")

        return df

    def final_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs final checks, like ensuring no NaNs remain and selecting columns."""
        logger.debug("Performing final checks...")

        # Check for remaining NaNs
        nan_counts = df.isnull().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if not cols_with_nan.empty:
            logger.warning(f"NaN values still present after cleaning in columns: {cols_with_nan.to_dict()}. Attempting final fill with 0.")
            # Consider a more robust final fill strategy if needed (e.g., ffill/bfill)
            df = df.fillna(0)

        # Ensure all expected columns are present (fill with default if missing?)
        present_cols = set(df.columns)
        missing_expected = self.expected_columns - present_cols
        if missing_expected:
            logger.warning(f"Expected columns missing after cleaning: {missing_expected}. They might not be in raw data or were dropped.")
            # Optionally add missing columns with default values (e.g., 0)
            # for col in missing_expected:
            #     df[col] = 0

        # Select and reorder columns (optional, ensures consistency)
        # final_cols = [col for col in self.expected_columns if col in df.columns]
        # df = df[final_cols]

        return df


# --- Standalone Function (Less Recommended than Pipeline) ---
# Kept for backward compatibility or simple use cases
def clean_battery_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Applies a sequence of cleaning steps to the battery pack DataFrame.
    Consider using the DataCleaningPipeline class for better structure.
    """
    pipeline = DataCleaningPipeline(config)
    return pipeline.run(df)


# --- Example Usage ---
if __name__ == '__main__':
    print("Testing DataCleaningPipeline for Battery Pack Data...")

    # Create dummy config reflecting pack structure
    dummy_config = {
        'data': {
            'features': ['pack_voltage', 'pack_current', 'pack_temperature_avg', 'step_time_s'],
            'target': ['pack_soh'],
            'time_column': 'timestamp',
            'cycle_column': 'cycle_index',
            'step_column': 'step_identifier', # Using a different name to test mapping
            'voltage_range': [13 * 2.5, 13 * 4.2], # 32.5V - 54.6V
            'current_range': [-15, 10], # Example: -15A to +10A
            'temperature_range': [-10, 65],
            'charge_discharge_threshold': 0.2,
            'column_mapping': { # Example mapping
                'Voltage': 'pack_voltage',
                'Amps': 'pack_current',
                'Temp_avg': 'pack_temperature_avg',
                'DateTime': 'timestamp',
                'Cycle': 'cycle_index',
                'Step': 'step_identifier', # Raw name maps to config's step_column
                'Pack_SOH_Estimate': 'pack_soh'
            }
        }
        # Add other necessary config sections if needed by pipeline methods
    }

    # Create dummy DataFrame with pack-level names and potential issues
    data = {
        'DateTime': pd.to_datetime(['2025-04-19 10:00:00', '2025-04-19 10:00:10', '2025-04-19 10:00:20',
                         '2025-04-19 10:00:30', '2025-04-19 10:00:40', '2025-04-19 10:00:50']),
        'Cycle': [1, 1, 1, 1, 2, 2],
        'Step': ['charge', 'charge', 'discharge', 'discharge', 'charge', 'charge'],
        'Voltage': [48.0, 48.5, 45.0, 44.5, 49.0, 60.0], # Includes voltage outlier and one outside range
        'Amps': [5.0, 5.1, -10.0, -9.9, 5.0, 5.0],
        'Temp_avg': [25.0, 25.5, 28.0, None, 26.0, 26.5], # Includes None
        'Pack_SOH_Estimate': [0.99, 0.99, 0.989, 0.989, 0.988, 0.988],
        'Unmapped_Col': [1]*6
    }
    raw_df = pd.DataFrame(data)
    print("\n--- Original Dummy DataFrame ---")
    print(raw_df)
    print(raw_df.info())

    # Initialize and run the pipeline
    cleaner = DataCleaningPipeline(dummy_config)
    cleaned_df = cleaner.run(raw_df, filename="dummy_pack_data.csv")

    print("\n--- Cleaned DataFrame ---")
    print(cleaned_df)
    print(cleaned_df.info())

    # Check results
    assert 'pack_voltage' in cleaned_df.columns
    assert 'step_type' in cleaned_df.columns # Check mapping worked
    assert not cleaned_df['pack_temperature_avg'].isnull().any(), "NaNs not handled in temperature"
    assert cleaned_df['pack_voltage'].max() < 55.0, "Voltage outlier or out-of-range value not removed"
    assert 'Unmapped_Col' in cleaned_df.columns, "Unmapped column should still exist unless explicitly dropped"
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df['timestamp'])
    assert pd.api.types.is_categorical_dtype(cleaned_df['step_type'])

    print("\nDataCleaningPipeline test completed successfully.")