# battery_project/data/__init__.py

# Import core components from existing modules
from .cleaning import DataCleaner
from .preprocessing import DataProcessor
from .data_provider import OptimizedDataLoader, FormatParser, DataAugmentor

# --- Import components from NEW modules ---
# Import feature engineering functions/classes
from .feature_engineering import (
    extract_health_features,
    calculate_ic,
    calculate_dtv,
    calculate_ctv
)
# Import simulation interface functions/classes
from .simulation_interface import SimulationInterface, get_simulation_interface

# --- Define what is available for import using 'from battery_project.data import *' ---
# Or simply define what should be easily accessible from the package.
__all__ = [
    # Existing components
    "DataCleaner",
    "DataProcessor",
    "OptimizedDataLoader",
    "FormatParser",
    "DataAugmentor",
    # New feature engineering components
    "extract_health_features",
    "calculate_ic",
    "calculate_dtv",
    "calculate_ctv",
    # New simulation interface components
    "SimulationInterface",
    "get_simulation_interface",
]

# You can also perform package-level initialization here if needed
import logging
logger = logging.getLogger(__name__)
logger.debug("Battery project data package initialized.")