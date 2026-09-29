# Battery State-of-Health (SOH) Prediction with Unsupervised Transfer Learning

My Master's thesis project (National Central University, June 2025): predicting the State of Health (SOH) of lithium-ion batteries across operating temperatures using deep learning with unsupervised transfer learning.

## Overview

Battery SOH estimation degrades when a model trained at one temperature is applied to another, because charge–discharge characteristics shift with temperature (domain shift). This project builds a CNN+LSTM baseline model trained on 25°C charge–discharge data and adapts it to new temperature domains using **AdaBN (Adaptive Batch Normalization)** — an unsupervised transfer-learning technique that requires no target-domain labels and no backpropagation. The pipeline includes data preprocessing, model training with configurable trainers, and evaluation scripts.

## Method

- **Baseline model:** CNN+LSTM network that takes battery charge–discharge sequences and predicts SOH, trained on 25°C data.
- **Unsupervised transfer learning (AdaBN):** adapts the 25°C baseline to 5°C (and other temperature) data by aligning batch-normalization statistics with the target domain — using only unlabeled target data and no retraining by backpropagation.
- **Data:** in-house charge–discharge measurements collected at 25°C, 5°C, and 45°C, plus the open NASA battery dataset for cross-dataset validation.

## Results

| Setting | In-house data | NASA B0006 |
| --- | --- | --- |
| 25°C baseline applied directly to 5°C data (domain shift) | RMSE 15.64% | RMSE 11.68% |
| After AdaBN unsupervised transfer (no labels, no backprop) | RMSE **5.58%** | RMSE **5.01%** |

## Tech stack

- Python
- Keras (deep learning models)
- Scikit-learn (data preprocessing and metrics)

## Repository structure

```text
battery_project/
├── config/            # Base, system, and HPO search-space configs (base_config.py, system_config.py, hpo_search_space.py)
├── core/              # Logging and memory utilities (logging.py, memory.py)
├── data/              # BatteryDataProvider — dataset loading and batching
├── models/            # Model definitions and model registry
├── scripts/           # Runnable entry points: preprocess_data.py, clean_data.py, train.py, evaluate.py
├── trainers/          # Training logic: base_trainer.py, multitemp_trainer.py, gan_trainer.py,
│                      # pinn_trainer.py, losses.py, callbacks.py, distributed.py
├── utils/             # Shared helper utilities
└── logs/              # Training logs and outputs
```

## How to run

1. Install Python dependencies:

   ```bash
   pip install tensorflow keras scikit-learn
   ```

2. Preprocess the raw data:

   ```bash
   python scripts/preprocess_data.py
   ```

3. Train the baseline model (uses `config/base_config.py` by default):

   ```bash
   python scripts/train.py --config config/base_config.py
   ```

   Useful options: `--data-dir`, `--output-dir`, `--experiment-name`, `--model-name`, `--trainer-name`, and `--tune` for Ray Tune hyperparameter search.

4. Evaluate the trained model:

   ```bash
   python scripts/evaluate.py
   ```

## Notes

- The in-house dataset is not published here; the pipeline also supports the public NASA battery dataset format.
