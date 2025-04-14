# battery_project/data/feature_engineering.py

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter # Keep for potential future use (e.g., smoothing rates)
from scipy.stats import skew, kurtosis
# from scipy.signal import find_peaks # Removed as _extract_peaks is commented out
# from scipy.interpolate import interp1d # Removed as _interpolate_curve is commented out

import logging
from typing import List, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# --- Constants for Feature Extraction ---
# Keep Savgol params if smoothing is needed elsewhere
SMOOTH_WINDOW_LENGTH = 11 # Must be odd
SMOOTH_POLYORDER = 2

# Resistance Estimation Params
RESISTANCE_WINDOW = 5 # Number of points to average voltage/current change over during step transition
RESISTANCE_CURRENT_THRESH = 0.5 # Minimum current change to trigger resistance estimation (Amps) - Adjust based on pack

# --- Commented Out Single-Cell IC/DTV Functions ---
# NOTE: Applying IC (dQ/dV) or DTV (dT/dV) analysis directly to PACK voltage (V_pack)
# is generally unreliable. Peaks from individual cells overlap and smooth out,
# making interpretation difficult and often meaningless without deconvolution techniques.
# These functions are kept commented out for reference but are NOT used in the
# pack-level feature extraction below.

# def _smooth_diff(y: np.ndarray, x: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
#     # ... (Implementation remains the same if needed elsewhere) ...
#     pass

# def _extract_peaks(y: np.ndarray, x: np.ndarray, prominence: float, distance: int) -> Dict[str, float]:
#    # ... (Implementation remains the same if needed elsewhere) ...
#    pass

# def _interpolate_curve(x: np.ndarray, y: np.ndarray, x_interp: np.ndarray) -> Optional[np.ndarray]:
#    # ... (Implementation remains the same if needed elsewhere) ...
#    pass

# def calculate_ic(voltage: np.ndarray, capacity: np.ndarray) -> Dict[str, float]:
#    """
#    Calculates Incremental Capacity (dQ/dV) features.
#    NOTE: Not suitable for direct application to pack voltage.
#    """
#    # ... (Implementation remains the same if needed elsewhere) ...
#    logger.warning("IC calculation is generally not suitable for pack-level voltage.")
#    return {}

# def calculate_dtv(voltage: np.ndarray, temperature: np.ndarray) -> Dict[str, float]:
#    """
#    Calculates Differential Thermal Voltammetry (dT/dV) features.
#    NOTE: Not suitable for direct application to pack voltage.
#    """
#    # ... (Implementation remains the same if needed elsewhere) ...
#    logger.warning("DTV calculation is generally not suitable for pack-level voltage.")
#    return {}


# --- Pack-Level Feature Calculation Functions ---

def calculate_ctv(cycle_data: pd.DataFrame, prev_cycle_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Calculates Capacity Temperature Variation (CTV) features for the PACK.
    Defined here as changes in pack capacity correlated with average pack
    temperature changes between cycles.

    Args:
        cycle_data (pd.DataFrame): Data for the current cycle, must contain
                                   'charge_capacity', 'discharge_capacity',
                                   'pack_temperature_avg'.
        prev_cycle_data (Optional[pd.DataFrame]): Data for the previous cycle.

    Returns:
        Dict[str, float]: Dictionary containing CTV features for the pack.
    """
    features = {}
    # Use pack-level column names (adjust if different in your config/data)
    charge_cap_col = 'charge_capacity'
    discharge_cap_col = 'discharge_capacity'
    temp_col = 'pack_temperature_avg'

    required_cols = [charge_cap_col, discharge_cap_col, temp_col]
    if not all(col in cycle_data.columns for col in required_cols):
        logger.warning(f"Missing required columns for Pack CTV in current cycle: {required_cols}")
        return features

    # Calculate based on the *final* capacity values in the cycle data
    current_charge_cap = cycle_data[charge_cap_col].iloc[-1] if not cycle_data[charge_cap_col].empty else 0
    current_discharge_cap = cycle_data[discharge_cap_col].iloc[-1] if not cycle_data[discharge_cap_col].empty else 0
    current_avg_temp = cycle_data[temp_col].mean() # Average temp over the cycle

    features['ctv_pack_charge_capacity'] = current_charge_cap
    features['ctv_pack_discharge_capacity'] = current_discharge_cap
    features['ctv_pack_avg_temp'] = current_avg_temp

    if prev_cycle_data is not None and all(col in prev_cycle_data.columns for col in required_cols):
        prev_charge_cap = prev_cycle_data[charge_cap_col].iloc[-1] if not prev_cycle_data[charge_cap_col].empty else 0
        prev_discharge_cap = prev_cycle_data[discharge_cap_col].iloc[-1] if not prev_cycle_data[discharge_cap_col].empty else 0
        prev_avg_temp = prev_cycle_data[temp_col].mean()

        delta_temp = current_avg_temp - prev_avg_temp
        delta_charge_cap = current_charge_cap - prev_charge_cap
        delta_discharge_cap = current_discharge_cap - prev_discharge_cap

        features['ctv_delta_temp_pack'] = delta_temp
        features['ctv_delta_charge_cap_pack'] = delta_charge_cap
        features['ctv_delta_discharge_cap_pack'] = delta_discharge_cap

        # Calculate capacity change per degree Celsius change (simple linear estimate)
        if abs(delta_temp) > 0.1: # Avoid division by near zero
            features['ctv_charge_cap_per_delta_temp_pack'] = delta_charge_cap / delta_temp
            features['ctv_discharge_cap_per_delta_temp_pack'] = delta_discharge_cap / delta_temp
        else:
            features['ctv_charge_cap_per_delta_temp_pack'] = 0.0
            features['ctv_discharge_cap_per_delta_temp_pack'] = 0.0

    return features

def calculate_coulombic_efficiency(cycle_data: pd.DataFrame) -> Dict[str, float]:
    """Calculates the coulombic efficiency for the pack."""
    features = {}
    charge_cap_col = 'charge_capacity'
    discharge_cap_col = 'discharge_capacity'

    if charge_cap_col not in cycle_data.columns or discharge_cap_col not in cycle_data.columns:
        logger.warning(f"Missing capacity columns for Coulombic Efficiency: {charge_cap_col}, {discharge_cap_col}")
        return {'pack_coulombic_efficiency': 0.0} # Return default

    # Use final values in the cycle
    charge_cap = cycle_data[charge_cap_col].iloc[-1] if not cycle_data[charge_cap_col].empty else 0
    discharge_cap = cycle_data[discharge_cap_col].iloc[-1] if not cycle_data[discharge_cap_col].empty else 0

    if charge_cap > 1e-6: # Avoid division by zero
        ce = discharge_cap / charge_cap
        # Clamp CE between reasonable bounds (e.g., 0.8 and 1.1)
        features['pack_coulombic_efficiency'] = np.clip(ce, 0.8, 1.1)
    else:
        features['pack_coulombic_efficiency'] = 0.0

    return features


def estimate_pack_dc_resistance(cycle_data: pd.DataFrame, window: int = RESISTANCE_WINDOW, current_thresh: float = RESISTANCE_CURRENT_THRESH) -> Dict[str, float]:
    """
    Estimates pack DC resistance by looking at voltage changes during current step changes.
    This is a simplified estimation.
    """
    features = {}
    voltage_col = 'pack_voltage'
    current_col = 'pack_current'
    step_col = 'step_type' # Assumes step_type is marked ('charge', 'discharge', 'rest')

    if not all(col in cycle_data.columns for col in [voltage_col, current_col, step_col]):
        logger.warning(f"Missing columns for Pack DC Resistance estimation: {voltage_col}, {current_col}, {step_col}")
        return {'pack_dc_resistance_charge_est': 0.0, 'pack_dc_resistance_discharge_est': 0.0}

    # Find transitions between steps (e.g., rest-to-charge, charge-to-rest, rest-to-discharge)
    step_changes = cycle_data[step_col].diff().fillna(method='bfill') != 0
    change_indices = cycle_data.index[step_changes]

    resistances_charge = []
    resistances_discharge = []

    for idx in change_indices:
        if idx < window or idx >= len(cycle_data) - window: # Ensure enough points around change
            continue

        # Look at window before and after the change index 'idx'
        # This logic assumes the change happens *at* idx
        v_before = cycle_data[voltage_col].iloc[idx - window : idx].mean()
        i_before = cycle_data[current_col].iloc[idx - window : idx].mean()
        v_after = cycle_data[voltage_col].iloc[idx : idx + window].mean()
        i_after = cycle_data[current_col].iloc[idx : idx + window].mean()

        delta_v = v_after - v_before
        delta_i = i_after - i_before

        # Estimate resistance if current change is significant
        if abs(delta_i) > current_thresh:
            resistance_est = delta_v / delta_i
            # Check if it's a charge or discharge transition based on current after change
            current_step_type = cycle_data[step_col].iloc[idx] # Step type after transition
            if current_step_type == 'charge' and resistance_est > 0: # Expect positive resistance
                 resistances_charge.append(resistance_est)
            elif current_step_type == 'discharge' and resistance_est > 0: # Expect positive resistance
                 resistances_discharge.append(resistance_est)
            # Ignore negative resistance estimates (likely noise or complex effects)

    # Average the estimated resistances (or use median for robustness)
    if resistances_charge:
        features['pack_dc_resistance_charge_est'] = np.median(resistances_charge)
    else:
        features['pack_dc_resistance_charge_est'] = 0.0 # Or NaN?

    if resistances_discharge:
        features['pack_dc_resistance_discharge_est'] = np.median(resistances_discharge)
    else:
        features['pack_dc_resistance_discharge_est'] = 0.0 # Or NaN?

    # Clamp resistance to reasonable bounds (e.g., 1 mOhm to 1 Ohm for a pack)
    features['pack_dc_resistance_charge_est'] = np.clip(features['pack_dc_resistance_charge_est'], 0.001, 1.0)
    features['pack_dc_resistance_discharge_est'] = np.clip(features['pack_dc_resistance_discharge_est'], 0.001, 1.0)


    return features

def calculate_voltage_curve_stats(cycle_data: pd.DataFrame) -> Dict[str, float]:
    """Calculates statistics on the pack voltage curve during charge/discharge."""
    features = {}
    voltage_col = 'pack_voltage'
    time_col = 'step_time_s' # Use time within step
    step_col = 'step_type'

    if not all(col in cycle_data.columns for col in [voltage_col, time_col, step_col]):
        logger.warning(f"Missing columns for Voltage Curve Stats: {voltage_col}, {time_col}, {step_col}")
        return {}

    for step_type in ['charge', 'discharge']:
        step_df = cycle_data[cycle_data[step_col] == step_type]
        if len(step_df) > 10: # Need sufficient points
            voltage = step_df[voltage_col].values
            time = step_df[time_col].values

            features[f'voltage_mean_{step_type}'] = np.mean(voltage)
            features[f'voltage_variance_{step_type}'] = np.var(voltage)
            features[f'voltage_skewness_{step_type}'] = skew(voltage)
            features[f'voltage_kurtosis_{step_type}'] = kurtosis(voltage)

            # Calculate average slope (dV/dt) - simple estimate
            voltage_smooth = savgol_filter(voltage, window_length=SMOOTH_WINDOW_LENGTH, polyorder=SMOOTH_POLYORDER, mode='nearest')
            dv_dt = np.gradient(voltage_smooth, time)
            dv_dt = np.nan_to_num(dv_dt) # Handle potential NaNs
            features[f'dv_dt_mean_{step_type}'] = np.mean(dv_dt)
            features[f'dv_dt_variance_{step_type}'] = np.var(dv_dt)
        else:
            # Fill with defaults if step has too few points
            for stat in ['mean', 'variance', 'skewness', 'kurtosis']:
                 features[f'voltage_{stat}_{step_type}'] = 0.0
            for stat in ['mean', 'variance']:
                 features[f'dv_dt_{stat}_{step_type}'] = 0.0

    return features


def calculate_temperature_change(cycle_data: pd.DataFrame) -> Dict[str, float]:
    """Calculates the change in average pack temperature during charge/discharge."""
    features = {}
    temp_col = 'pack_temperature_avg'
    step_col = 'step_type'

    if not all(col in cycle_data.columns for col in [temp_col, step_col]):
        logger.warning(f"Missing columns for Temperature Change: {temp_col}, {step_col}")
        return {}

    for step_type in ['charge', 'discharge']:
        step_df = cycle_data[cycle_data[step_col] == step_type]
        if len(step_df) > 1:
             temp_start = step_df[temp_col].iloc[0]
             temp_end = step_df[temp_col].iloc[-1]
             features[f'delta_temp_{step_type}'] = temp_end - temp_start
        else:
             features[f'delta_temp_{step_type}'] = 0.0

    return features


# --- Orchestrator ---

def extract_health_features(
    cycle_data: pd.DataFrame,
    cycle_number: int,
    prev_cycle_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None # Pass config if needed for parameters
    ) -> Dict[str, float]:
    """
    Extracts various health indicator features from a single cycle's PACK data.
    Focuses on pack-level characteristics instead of single-cell IC/DTV.

    Args:
        cycle_data (pd.DataFrame): DataFrame containing data for one cycle.
                                   Expected columns: 'pack_voltage', 'pack_current',
                                   'pack_temperature_avg', 'charge_capacity',
                                   'discharge_capacity', 'step_type', 'step_time_s'.
        cycle_number (int): The current cycle number.
        prev_cycle_data (Optional[pd.DataFrame]): Data for the previous cycle, for CTV calculation.
        config (Optional[Dict]): Configuration dictionary (potentially for parameters).

    Returns:
        Dict[str, float]: Dictionary of extracted features for this cycle.
    """
    logger.info(f"Extracting PACK features for cycle {cycle_number}...")
    all_features = {'cycle_number': float(cycle_number)}

    # --- TODO: Handle Individual Cell Data (If Available) ---
    # If cycle_data contains individual cell measurements (e.g., 'cell_1_voltage', 'cell_13_temp'):
    # - Calculate statistics across cells: std dev, min, max, range for voltage/temp.
    # - Potentially calculate features for each cell and aggregate (mean, median, min/max).
    # - Add these derived features to the 'all_features' dict.
    # Example:
    # num_cells = config.get('pack_config', {}).get('num_cells_series', 1)
    # cell_volt_cols = [f'cell_{i}_voltage' for i in range(1, num_cells + 1) if f'cell_{i}_voltage' in cycle_data.columns]
    # if cell_volt_cols:
    #     all_features['voltage_std_dev_cells'] = cycle_data[cell_volt_cols].std(axis=1).mean() # Avg std dev over cycle
    #     all_features['voltage_range_cells'] = (cycle_data[cell_volt_cols].max(axis=1) - cycle_data[cell_volt_cols].min(axis=1)).mean()
    # Similarly for temperature variance etc.

    if cycle_data.empty:
        logger.warning(f"Cycle {cycle_number} data is empty. Cannot extract features.")
        return all_features

    # --- Calculate Pack Coulombic Efficiency ---
    try:
        ce_features = calculate_coulombic_efficiency(cycle_data)
        all_features.update(ce_features)
    except Exception as e:
        logger.error(f"Error calculating Coulombic Efficiency for cycle {cycle_number}: {e}", exc_info=True)

    # --- Calculate Pack DC Resistance ---
    try:
        res_features = estimate_pack_dc_resistance(cycle_data)
        all_features.update(res_features)
    except Exception as e:
        logger.error(f"Error estimating DC Resistance for cycle {cycle_number}: {e}", exc_info=True)

    # --- Calculate Voltage Curve Stats ---
    try:
        v_stats_features = calculate_voltage_curve_stats(cycle_data)
        all_features.update(v_stats_features)
    except Exception as e:
        logger.error(f"Error calculating Voltage Curve Stats for cycle {cycle_number}: {e}", exc_info=True)

    # --- Calculate Temperature Changes ---
    try:
        t_change_features = calculate_temperature_change(cycle_data)
        all_features.update(t_change_features)
    except Exception as e:
        logger.error(f"Error calculating Temperature Change for cycle {cycle_number}: {e}", exc_info=True)

    # --- Calculate Pack CTV Features ---
    try:
        ctv_features = calculate_ctv(cycle_data, prev_cycle_data)
        all_features.update(ctv_features)
    except Exception as e:
         logger.error(f"Error calculating Pack CTV for cycle {cycle_number}: {e}", exc_info=True)

    # --- Add other simple features ---
    # Basic capacity and duration features (already present in CTV, but can be added directly)
    all_features['end_charge_capacity_pack'] = cycle_data['charge_capacity'].iloc[-1] if 'charge_capacity' in cycle_data.columns and not cycle_data['charge_capacity'].empty else 0.0
    all_features['end_discharge_capacity_pack'] = cycle_data['discharge_capacity'].iloc[-1] if 'discharge_capacity' in cycle_data.columns and not cycle_data['discharge_capacity'].empty else 0.0

    if 'total_time_s' in cycle_data.columns and not cycle_data['total_time_s'].empty:
        all_features['cycle_duration_s'] = cycle_data['total_time_s'].iloc[-1] - cycle_data['total_time_s'].iloc[0]
    elif 'step_time_s' in cycle_data.columns and not cycle_data['step_time_s'].empty:
         # Approximate using step time if total time is missing
         all_features['cycle_duration_s'] = cycle_data['step_time_s'].iloc[-1] # This assumes steps cover whole cycle
    else:
        all_features['cycle_duration_s'] = 0.0


    # --- Remove IC/DTV Feature Extraction Calls ---
    # logger.debug(f"Skipping IC features for cycle {cycle_number} (Pack data)")
    # logger.debug(f"Skipping DTV features for cycle {cycle_number} (Pack data)")

    logger.info(f"Finished extracting features for pack cycle {cycle_number}. Found {len(all_features)} features.")
    # Ensure all features are float type for consistency
    return {k: float(v) if pd.notna(v) and isinstance(v, (int, float, np.number)) else 0.0 for k, v in all_features.items()}


# Example Usage (within data/preprocessing.py)
if __name__ == '__main__':
    print("Testing Pack-Level Feature Extraction...")

    # Create dummy cycle data (pack level)
    cycle_data_1 = pd.DataFrame({
        'charge_capacity': np.linspace(0, 6.2, 50),
        'discharge_capacity': np.linspace(0, 6.0, 50),
        'pack_temperature_avg': np.random.normal(25, 1, 50),
        'step_type': ['charge']*25 + ['discharge']*25,
        'pack_voltage': np.concatenate([np.linspace(35.0, 54.0, 25), np.linspace(54.0, 33.0, 25)]),
        'pack_current': np.concatenate([np.full(25, 5.0), np.full(25, -10.0)]),
        'step_time_s': np.concatenate([np.linspace(0, 3600, 25), np.linspace(0, 1800, 25)]),
        'total_time_s': np.linspace(0, 5400, 50)
    })
    # Add step transitions for resistance test
    cycle_data_1['pack_voltage'].iloc[24] = 54.2 # End charge higher
    cycle_data_1['pack_voltage'].iloc[25] = 53.8 # Start discharge lower
    cycle_data_1['pack_current'].iloc[24] = 4.9 # End charge lower current
    cycle_data_1['pack_current'].iloc[25] = -10.1 # Start discharge higher current


    cycle_data_2 = pd.DataFrame({
        'charge_capacity': np.linspace(0, 6.1, 50),
        'discharge_capacity': np.linspace(0, 5.9, 50),
        'pack_temperature_avg': np.random.normal(28, 1, 50), # Higher avg temp
        'step_type': ['charge']*25 + ['discharge']*25,
        'pack_voltage': np.concatenate([np.linspace(35.0, 54.0, 25), np.linspace(54.0, 33.0, 25)]),
        'pack_current': np.concatenate([np.full(25, 5.0), np.full(25, -10.0)]),
        'step_time_s': np.concatenate([np.linspace(0, 3500, 25), np.linspace(0, 1750, 25)]),
        'total_time_s': np.linspace(5500, 10750, 50)
    })
    cycle_data_2['pack_voltage'].iloc[24] = 54.1
    cycle_data_2['pack_voltage'].iloc[25] = 53.7


    # Test features for cycle 2, using cycle 1 as previous
    print("\n--- Features for Cycle 2 ---")
    all_f = extract_health_features(cycle_data=cycle_data_2, cycle_number=2, prev_cycle_data=cycle_data_1)
    print(all_f)

    # Basic checks
    assert 'cycle_number' in all_f
    assert 'pack_coulombic_efficiency' in all_f
    assert 'pack_dc_resistance_discharge_est' in all_f
    assert 'voltage_mean_charge' in all_f
    assert 'delta_temp_discharge' in all_f
    assert 'ctv_delta_discharge_cap_pack' in all_f
    assert 'ic_num_peaks' not in all_f # Ensure IC features are not generated
    assert 'dtv_num_peaks' not in all_f # Ensure DTV features are not generated
    assert all_f['pack_dc_resistance_discharge_est'] > 0.0 # Resistance should be positive
    assert 0.8 < all_f['pack_coulombic_efficiency'] < 1.1 # CE should be in reasonable range

    print("\nPack-Level Feature Extraction Test completed successfully.")