# New File: utils/environment.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment Validation and Setup Utilities.
"""
import os
import sys
import platform
import logging
import random
import numpy as np
import tensorflow as tf
import psutil

# --- Logging Setup ---
# (Assume LoggerFactory is available)
try:
    from core.logging import LoggerFactory
    logger = LoggerFactory.get_logger("utils.environment")
except ImportError: # pragma: no cover
    logger = logging.getLogger("utils.environment")
    if not logger.handlers: logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)

def set_random_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Optional: Configure TensorFlow determinism (might impact performance)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    logger.info(f"Set random seed to {seed} for Python, NumPy, and TensorFlow.")

def validate_environment(gpu_required: bool = False) -> bool:
    """Validates the Python and ML environment."""
    logger.info("===== Environment Validation =====")
    valid = True
    try:
        # Python Version
        py_version = platform.python_version()
        logger.info(f"Python Version: {py_version}")
        # Recommended: 3.8+
        if float(platform.python_version_tuple()[0] + '.' + platform.python_version_tuple()[1]) < 3.8:
             logger.warning("Python 3.8+ is recommended.")

        # TensorFlow Version
        tf_version = tf.__version__
        logger.info(f"TensorFlow Version: {tf_version}")
        # Recommended: 2.8+
        if float(tf_version.split('.')[0] + '.' + tf_version.split('.')[1]) < 2.8:
             logger.warning("TensorFlow 2.8+ is recommended.")

        # GPU Availability and Setup
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"Detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu.name}")
                try: tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError: logger.debug(f"Memory growth likely already set for GPU {i}.")
        elif gpu_required:
            logger.error("GPU is required for this configuration but none was detected!")
            valid = False
        else:
            logger.warning("No GPU detected. Training will run on CPU.")

        # CPU Info
        cpu_logical = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        logger.info(f"CPU: {cpu_physical} Physical Cores, {cpu_logical} Logical Processors")

        # Memory Info
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        avail_gb = mem.available / (1024**3)
        logger.info(f"System Memory: {total_gb:.2f} GB Total, {avail_gb:.2f} GB Available")
        if total_gb < 8:
             logger.warning("System memory is less than 8GB, which might be insufficient.")

        logger.info("==================================")

    except Exception as e:
        logger.error(f"Environment validation failed: {e}", exc_info=True)
        valid = False

    return valid