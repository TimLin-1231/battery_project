# battery_project/data/simulation_interface.py

import logging
from typing import Optional, Dict, List
import time
import os

logger = logging.getLogger(__name__)

# --- Optional PyBaMM Import ---
# Wrap in try-except to make PyBaMM an optional dependency.
# Set environment variable to potentially disable PyBaMM logging if too verbose.
# os.environ['PYBAMM_LOG_LEVEL'] = 'WARNING' # Example: Set log level
try:
    import pybamm
    import pandas as pd
    import numpy as np
    PYBAMM_AVAILABLE = True
    logger.info("PyBaMM library found and imported successfully.")
except ImportError:
    PYBAMM_AVAILABLE = False
    pd = None # Make pandas unavailable if pybamm isn't
    np = None
    pybamm = None # Define pybamm as None to avoid NameError later
    logger.warning("PyBaMM library not found. SimulationInterface functionality will be limited.")
except Exception as e:
    # Catch other potential import errors
    PYBAMM_AVAILABLE = False
    pd = None
    np = None
    pybamm = None
    logger.error(f"An error occurred during PyBaMM import: {e}", exc_info=True)


class SimulationInterface:
    """
    Interface for interacting with battery simulation tools like PyBaMM.
    Designed to generate data for surrogate models or validation.
    Can be configured to use different PyBaMM models, parameters, and solvers.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the simulation interface based on the provided configuration.

        Args:
            config: Configuration dictionary, expected to contain simulator settings
                    under the 'simulator' key (see config/base_config.py).
        """
        self.config = config or {}
        sim_config = self.config.get("simulator", {})
        self.is_enabled = sim_config.get("enabled", False)
        self.simulator_type = sim_config.get("type", "pybamm" if PYBAMM_AVAILABLE else "none")

        self.model = None
        self.solver = None
        self.parameter_values = None
        self._is_initialized = False

        if self.is_enabled and self.simulator_type == "pybamm" and PYBAMM_AVAILABLE:
            self._init_pybamm(sim_config.get("pybamm", {}))
        elif self.is_enabled and not PYBAMM_AVAILABLE:
            logger.error("Simulation is enabled in config, but PyBaMM library is not available.")
            self.is_enabled = False # Disable if library missing
        elif not self.is_enabled:
             logger.info("Simulation interface is disabled in configuration.")
        else:
            logger.warning(f"Unsupported simulator type configured: {self.simulator_type}")
            self.is_enabled = False

    def _init_pybamm(self, pybamm_config: Dict):
        """Initializes the PyBaMM model, parameters, and solver based on config."""
        if not PYBAMM_AVAILABLE: return # Should not happen if check is done before calling

        logger.info("Initializing PyBaMM interface...")
        model_options = pybamm_config.get("model_options", None)
        solver_options = pybamm_config.get("solver_options", {"mode": "safe"})
        parameter_set_name = pybamm_config.get("parameter_set", "Chen2020")
        model_name = pybamm_config.get("model", "DFN")
        solver_name = pybamm_config.get("solver", "CasadiSolver")

        try:
            # --- Select PyBaMM Model ---
            if model_name == "DFN":
                 self.model = pybamm.lithium_ion.DFN(model_options)
            elif model_name == "SPM":
                 self.model = pybamm.lithium_ion.SPM(model_options)
            elif model_name == "SPMe":
                 self.model = pybamm.lithium_ion.SPMe(model_options)
            # Add more models like P2D (which is an alias for DFN currently)
            # elif model_name == "P2D":
            #    self.model = pybamm.lithium_ion.DFN(model_options) # P2D often refers to DFN
            else:
                 raise ValueError(f"Unsupported PyBaMM model specified: {model_name}")
            logger.info(f"PyBaMM model set to: {model_name}")

            # --- Load Parameter Set ---
            # Parameter sets can be strings (built-in) or dictionaries
            if isinstance(parameter_set_name, str):
                 logger.info(f"Loading PyBaMM parameter set: {parameter_set_name}")
                 self.parameter_values = pybamm.ParameterValues(parameter_set_name)
            elif isinstance(parameter_set_name, dict):
                 logger.info("Loading PyBaMM parameters from dictionary.")
                 self.parameter_values = pybamm.ParameterValues(parameter_set_name)
            else:
                 raise ValueError(f"Invalid parameter_set format: {type(parameter_set_name)}")

            # --- Select Solver ---
            if solver_name == "CasadiSolver":
                 self.solver = pybamm.CasadiSolver(**solver_options)
            elif solver_name == "IDAKLUSolver":
                 # IDAKLU might need specific setup or dependencies
                 self.solver = pybamm.IDAKLUSolver(**solver_options)
            elif solver_name == "ScipySolver":
                 self.solver = pybamm.ScipySolver(**solver_options)
            else:
                 raise ValueError(f"Unsupported PyBaMM solver specified: {solver_name}")
            logger.info(f"PyBaMM solver set to: {solver_name} with options {solver_options}")

            self._is_initialized = True
            logger.info("PyBaMM interface initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize PyBaMM components: {e}", exc_info=True)
            self.model = None
            self.solver = None
            self.parameter_values = None
            self.is_enabled = False # Disable if initialization fails
            self._is_initialized = False

    def is_active(self) -> bool:
        """Checks if the simulation interface is enabled and successfully initialized."""
        return self.is_enabled and self._is_initialized

    def simulate_experiment(
        self,
        experiment_definition: Union[List[str], List[Tuple]], # PyBaMM Experiment format
        initial_soc: float = 1.0,
        parameter_overrides: Optional[Dict] = None, # Allow overriding specific parameters per simulation
        output_variables: Optional[List[str]] = None,
        t_eval: Optional[np.ndarray] = None # Optional time points for evaluation
    ) -> Optional[pd.DataFrame]:
        """
        Simulates a battery experiment defined using PyBaMM's Experiment class.

        Args:
            experiment_definition: A list of strings or tuples defining the steps,
                                   compatible with pybamm.Experiment().
            initial_soc: Initial State of Charge (0 to 1).
            parameter_overrides: Dictionary of parameters to temporarily override
                                 for this simulation (e.g., {"Ambient temperature [K]": 298.15}).
            output_variables: List of variables to output from the simulation. If None,
                              uses a default set.
            t_eval: Optional array of time points [seconds] at which to store the solution.

        Returns:
            A pandas DataFrame containing the simulation results, or None if failed.
        """
        if not self.is_active():
            logger.error("Cannot simulate: Simulator is not active or available.")
            return None

        if output_variables is None:
            output_variables = [
                "Time [s]", "Voltage [V]", "Current [A]", "Terminal voltage [V]",
                "Temperature [C]", "Discharge capacity [A.h]", "Charge capacity [A.h]",
                "State of Charge" # Example common variables
            ]
            # Add other potentially interesting variables:
            # "Electrolyte concentration [mol.m-3]", "Negative particle surface concentration [mol.m-3]", etc.

        logger.info(f"Starting PyBaMM experiment simulation...")
        logger.debug(f"Experiment Definition: {experiment_definition}")
        logger.debug(f"Initial SoC: {initial_soc}, Parameter Overrides: {parameter_overrides}")
        start_time = time.time()

        try:
            # --- Create a copy of parameters to avoid modifying the base set ---
            sim_params = self.parameter_values.copy()
            if parameter_overrides:
                 logger.info(f"Applying parameter overrides: {parameter_overrides}")
                 sim_params.update(parameter_overrides)

            # --- Define the experiment ---
            experiment = pybamm.Experiment(experiment_definition)

            # --- Create and run the simulation ---
            sim = pybamm.Simulation(
                 model=self.model,
                 experiment=experiment,
                 parameter_values=sim_params,
                 solver=self.solver
            )
            solution = sim.solve(initial_soc=initial_soc, t_eval=t_eval)

            # --- Extract data ---
            sim_df = self.extract_data(solution, output_variables)

            elapsed = time.time() - start_time
            if sim_df is not None:
                 logger.info(f"Simulation finished in {elapsed:.2f} seconds. Output shape: {sim_df.shape}")
            else:
                 logger.warning(f"Simulation finished in {elapsed:.2f} seconds, but data extraction failed.")

            return sim_df

        except Exception as e:
            logger.error(f"PyBaMM simulation failed: {e}", exc_info=True)
            return None

    def extract_data(self, solution, variables: List[str]) -> Optional[pd.DataFrame]:
        """
        Safely extracts specified variables from the PyBaMM solution object
        into a pandas DataFrame. Handles potential errors and shape mismatches.

        Args:
            solution: The solution object returned by sim.solve().
            variables: List of variable names (strings) to extract.

        Returns:
            A pandas DataFrame or None if extraction fails critically.
        """
        if not self.is_active() or solution is None:
             logger.error("Cannot extract data: Simulator inactive or solution is None.")
             return None
        if not pd: # Check if pandas is available
             logger.error("Cannot extract data: pandas library is not available.")
             return None

        logger.debug(f"Attempting to extract variables: {variables}")
        data = {}
        extracted_time = None

        # Try to get time first, as it's the base for length checks/interpolation
        time_keys = ["Time [s]", "Time [h]", "Time [min]"] # Common time keys
        for t_key in time_keys:
             if t_key in solution.t:
                 extracted_time = solution.t # Time is usually directly on solution
                 data['time_s'] = extracted_time # Store as seconds
                 # Convert if needed, e.g., if key was Time [h] -> data['time_s'] = extracted_time * 3600
                 logger.debug(f"Using time key: {t_key}, length: {len(extracted_time)}")
                 break
        if extracted_time is None or len(extracted_time) == 0:
             logger.error("Could not extract valid time vector from simulation solution.")
             return None

        # Extract each requested variable
        for var in variables:
            # Skip time if already extracted
            if var in time_keys and 'time_s' in data: continue

            try:
                var_data = solution[var] # Access variable data

                # Check if data needs processing (e.g., time-series extraction)
                if isinstance(var_data, pybamm.ProcessedVariable):
                    # Evaluate the variable at the solution time points
                    var_entries = var_data(t=extracted_time)
                elif isinstance(var_data, (np.ndarray, list)):
                     # Assume it's already evaluated if numpy array or list
                     var_entries = np.asarray(var_data)
                else:
                     logger.warning(f"Variable '{var}' has unexpected type {type(var_data)}. Attempting direct access.")
                     # Fallback attempt - might fail
                     var_entries = var_data.entries if hasattr(var_data, 'entries') else np.array([])


                # Handle potential spatial dimensions (e.g., concentration profiles)
                if len(var_entries.shape) > 1:
                     # Example: Average over spatial dimension(s) - adjust if needed
                     # Could also take surface value, center value, etc.
                     logger.debug(f"Variable '{var}' has shape {var_entries.shape}. Averaging over non-time dimensions.")
                     # Assume time is the last dimension if spatial dims come first
                     if var_entries.shape[-1] == len(extracted_time):
                          axes_to_avg = tuple(range(len(var_entries.shape) - 1))
                          var_entries = np.mean(var_entries, axis=axes_to_avg)
                     else: # Assume time is the first dimension
                          axes_to_avg = tuple(range(1, len(var_entries.shape)))
                          var_entries = np.mean(var_entries, axis=axes_to_avg)


                # Ensure length matches time after potential averaging
                if len(var_entries) != len(extracted_time):
                     logger.warning(f"Length mismatch for '{var}' ({len(var_entries)}) vs time ({len(extracted_time)}). Attempting interpolation.")
                     # Attempt interpolation using numpy.interp
                     # This assumes the variable's internal time scale matches the solution's time scale
                     # which might not always be true, especially with events.
                     try:
                         # Need an independent time variable for the variable itself if different from solution.t
                         # Assuming solution's time is the target
                         var_time = np.linspace(extracted_time[0], extracted_time[-1], len(var_entries)) # Assume linear time for var
                         var_entries = np.interp(extracted_time, var_time, var_entries)
                         logger.debug(f"Interpolation successful for '{var}'.")
                     except Exception as interp_err:
                         logger.error(f"Interpolation failed for '{var}': {interp_err}. Filling with NaN.")
                         var_entries = np.full_like(extracted_time, np.nan)
                else:
                     logger.debug(f"Successfully extracted '{var}' with matching length.")

                # Clean column name (remove units like [V], [A.h], etc.)
                clean_col_name = var.split(" [")[0].replace(" ", "_").lower()
                data[clean_col_name] = var_entries

            except KeyError:
                logger.warning(f"Variable '{var}' not found in the simulation solution. Skipping.")
            except Exception as e:
                logger.error(f"Error extracting or processing variable '{var}': {e}", exc_info=True)
                # Add column with NaNs to indicate failure for this variable
                clean_col_name = var.split(" [")[0].replace(" ", "_").lower()
                data[clean_col_name] = np.full_like(extracted_time, np.nan)

        # Convert collected data to DataFrame
        try:
            df = pd.DataFrame(data)
            # Ensure time column is named consistently if extracted differently
            if 'time_s' not in df.columns and extracted_time is not None:
                 df.insert(0, 'time_s', extracted_time)

            # Rename columns for consistency if needed
            df = df.rename(columns={
                "terminal_voltage": "voltage", # Example rename
                "state_of_charge": "soc"
            })
            logger.debug(f"Final extracted DataFrame columns: {df.columns.tolist()}")
            return df
        except Exception as df_err:
             logger.error(f"Failed to create pandas DataFrame from extracted data: {df_err}", exc_info=True)
             return None


# --- Helper function to get the interface ---
def get_simulation_interface(config: Dict) -> Optional[SimulationInterface]:
    """
    Factory function to create and return a SimulationInterface instance
    based on the global configuration.

    Args:
        config: The main configuration dictionary.

    Returns:
        An initialized SimulationInterface instance if enabled and available,
        otherwise None.
    """
    if config.get("simulator", {}).get("enabled", False):
        logger.info("Simulation interface requested by config.")
        interface = SimulationInterface(config)
        if interface.is_active():
             return interface
        else:
             logger.warning("Simulation interface was enabled but failed to activate (check logs).")
             return None
    else:
        logger.info("Simulation interface is disabled in config.")
        return None