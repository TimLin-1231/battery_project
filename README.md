# Multi-temperature battery SOH framework (2025 snapshot)

A modular TensorFlow 2 / Keras framework I built in April–May 2025, during my M.S. research at National Central University, Taiwan, for predicting the state of health (SOH) of Li-ion battery packs cycled at 5 °C, 25 °C and 45 °C. It combines CNN-LSTM baselines, a physics-informed pack model and GAN data augmentation, with a coordinator that trains a base model at one temperature and then transfers or fine-tunes it to the others.

> [!IMPORTANT]
> **Reference snapshot: not maintained, and it does not run as-is.**
> - This is a partial upload of a local working copy. `scripts/train.py` and `scripts/evaluate.py` import modules and helpers that are not in this repository (`core/config.py`, `core/system.py`, a `trainers` package with `TrainerFactory`, `utils/metrics.py`, `ensure_dir` in `utils/filesystem.py`), and `models/__init__.py` imports names that its modules do not define.
> - Some evaluation options are placeholders, for example cross-temperature evaluation and the uncertainty metrics in `scripts/evaluate.py`.
> - The battery data is not included.
>
> This framework is **not** the method of my M.S. thesis (2026). The thesis replaced it with a simpler CNN-LSTM model that is adapted across temperatures with AdaBN (label-free, no backprop); that code is not public.

## What's here

| Folder | Contents |
| --- | --- |
| `config/` | Base configuration, hardware/system profile, Ray Tune search space |
| `core/` | Logging and memory-management utilities |
| `data/` | Cleaning, feature engineering and preprocessing; data provider with noise and GAN augmentation; optional PyBaMM simulation interface |
| `models/` | CNN-LSTM and CNN-BiLSTM-attention baselines, pack-level physics-informed model (PINN) with RC-parameter layers, recurrent conditional GAN; attention and model-builder components |
| `trainers/` | Base, PINN and GAN (gradient-penalty) trainers; multi-temperature coordinator; multi-GPU training with `tf.distribute`; physics-informed losses and callbacks |
| `scripts/` | `clean_data.py`, `preprocess_data.py` (TFRecord output), `train.py` (with Ray Tune hyperparameter search), `evaluate.py` |
| `utils/` | Benchmarking, visualization, environment and file-system helpers |

## Tech stack

Python · TensorFlow 2 / Keras · Ray Tune · NumPy · pandas · scikit-learn · Matplotlib / Plotly
