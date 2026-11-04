# Battery SOH Prediction Framework: Early Thesis Prototype (2025)

English ｜ [繁體中文](README.zh-TW.md)

> **Status: archived prototype, not runnable as-is.**
> I built this experimental framework around April 2025, early in my Master's thesis at National Central University. It explores GAN-based data generation, physics-informed models and multi-temperature fine-tuning for lithium-ion battery-pack State of Health (SOH) prediction.
> **It is not the code behind my thesis results.** The final thesis method (CNN-LSTM + AdaBN) was developed later in a separate codebase that is not public. This repository contains no AdaBN code.

## What's in here

**Data pipeline**
- Cleaning of raw battery-management-system logs: timestamp repair, unit conversion, outlier handling (`scripts/clean_data.py`, `data/cleaning.py`).
- Preprocessing and leave-one-out cross-validation (LOOCV) splits (`scripts/preprocess_data.py`, `data/preprocessing.py`).
- Health-feature extraction: capacity–temperature variation, coulombic efficiency, pack DC resistance, voltage-curve statistics, temperature change (`data/feature_engineering.py`).
- Data loader with augmentation (`data/data_provider.py`), and an optional PyBaMM simulation interface (`data/simulation_interface.py`).

**Models** (`models/`)
- Baselines: CNN-LSTM and CNN-BiLSTM with attention.
- Recurrent conditional GAN (RCGAN) generator and discriminator for synthetic battery-pack time series.
- Physics-informed neural network (PINN) for a battery pack, based on an equivalent single-cell model, with RC-parameter and attention layers.

**Training** (`trainers/`)
- Base trainer, GAN trainer (with gradient penalty), PINN trainer with a physics-informed loss.
- Multi-temperature coordinator: trains a base model at one temperature, then transfers it to other temperatures by fine-tuning with configurable layer freezing.
- Distributed training (`tf.distribute`) and a Ray Tune hyperparameter search space (`config/hpo_search_space.py`).

**Evaluation and tooling**
- `scripts/evaluate.py`: metrics, SHAP analysis and uncertainty quantification.
- Hardware detection, memory management, benchmarking and plotting utilities (`config/system_config.py`, `core/`, `utils/`).

## Known issues

These are why the pipeline does not run end to end:

- `scripts/train.py`, `scripts/evaluate.py` and `trainers/gan_trainer.py` import modules that are not in this repository: `core.config`, `core.system` and `utils.metrics`.
- Import styles are mixed. Some modules expect the repository root on the path (`from config.base_config import ...`), some expect its parent (`from battery_project.core ...`), and some use relative imports (`from ..models ...`). No single way of launching the code satisfies all of them.
- `scripts/clean_data.py` imports `setup_logger` from `utils`, which does not provide it, so its project-logging block always falls back to standard logging. `DataCleaningPipeline.convert_units` is also defined twice in that file, and the later definition wins.
- There is no requirements file, no tests, and no data. The in-house measurements are not published.

## Relation to my thesis

My thesis (NCU, 2026) estimates battery-pack SOH across temperatures without target labels. It trains a CNN-LSTM model on 25 °C data and adapts it to 5 °C, 45 °C and the public NASA PCoE cells with AdaBN, which re-estimates only the BatchNorm statistics on unlabeled target data and needs no backpropagation. For example, RMSE goes from 15.64% to 5.58% on 5 °C pack #8, and from 11.68% to 5.01% on NASA cell B0006.

Those results come from the later codebase. They cannot be reproduced with the code in this repository.

## Tech stack

Python · TensorFlow/Keras · NumPy · pandas · SciPy · scikit-learn · Ray Tune · SHAP · PyBaMM (optional) · Matplotlib/Plotly

## Repository structure

```text
battery_project/
├── config/            # Base config, hardware detection (system_config.py), Ray Tune search space
├── core/              # Logging and memory management
├── data/              # Cleaning, preprocessing, feature engineering, data loader, PyBaMM interface
├── models/            # CNN-LSTM baselines, RCGAN, PINN, and shared layers (attention, RC parameters)
├── scripts/           # Entry points: clean_data.py, preprocess_data.py, train.py, evaluate.py
├── trainers/          # Base, GAN, PINN and multi-temperature trainers; losses; callbacks; distributed
└── utils/             # Benchmarking, file system, environment and visualization helpers
```
