# 電池健康狀態（SOH）預測框架：碩論早期原型（2025）

[English](README.md) ｜ 繁體中文

> **狀態：已封存的原型，目前無法直接執行。**
> 這是我在國立中央大學念碩士、論文研究初期（約 2025 年 4 月）寫的實驗框架，用來嘗試以 GAN 生成資料、物理資訊模型，以及跨溫度微調，來預測鋰離子電池組的健康狀態（State of Health，SOH）。
> **這裡不是產生碩論結果的程式碼。** 論文最終的方法（CNN-LSTM + AdaBN）是之後在另一份未公開的程式碼中完成的，本 repo 裡沒有任何 AdaBN 的程式。

## 內容

**資料流程**
- 清理電池管理系統（BMS）的原始紀錄：修正時間戳記、單位轉換、處理異常值（`scripts/clean_data.py`、`data/cleaning.py`）。
- 前處理，以及留一法交叉驗證（LOOCV）的資料切分（`scripts/preprocess_data.py`、`data/preprocessing.py`）。
- 健康特徵萃取：容量–溫度變化、庫倫效率、電池組直流內阻、電壓曲線統計量、溫度變化（`data/feature_engineering.py`）。
- 含資料擴增的資料載入器（`data/data_provider.py`），以及可選用的 PyBaMM 模擬介面（`data/simulation_interface.py`）。

**模型**（`models/`）
- 基準模型：CNN-LSTM，以及加上注意力機制的 CNN-BiLSTM。
- 遞迴條件式 GAN（RCGAN）的生成器與判別器，用來生成電池組的合成時間序列。
- 以等效單電芯模型為基礎的電池組物理資訊神經網路（PINN），搭配 RC 參數層與注意力層。

**訓練**（`trainers/`）
- 基礎訓練器、GAN 訓練器（含梯度懲罰）、使用物理資訊損失函數的 PINN 訓練器。
- 多溫度協調器：先在一個溫度訓練基礎模型，再以微調（可設定凍結哪些層）遷移到其他溫度。
- 分散式訓練（`tf.distribute`），以及 Ray Tune 的超參數搜尋空間（`config/hpo_search_space.py`）。

**評估與工具**
- `scripts/evaluate.py`：評估指標、SHAP 分析、不確定性量化。
- 硬體偵測、記憶體管理、效能基準測試與繪圖工具（`config/system_config.py`、`core/`、`utils/`）。

## 已知問題

以下是整個流程無法從頭跑到尾的原因：

- `scripts/train.py`、`scripts/evaluate.py` 和 `trainers/gan_trainer.py` import 了本 repo 裡不存在的模組：`core.config`、`core.system`、`utils.metrics`。
- import 的寫法不一致：有些模組要把 repo 根目錄加進路徑（`from config.base_config import ...`），有些要加它的上一層（`from battery_project.core ...`），有些用相對 import（`from ..models ...`），沒有一種啟動方式能同時滿足全部。
- `scripts/clean_data.py` 從 `utils` import `setup_logger`，但 `utils` 裡沒有這個函式，所以專案自訂的 logging 那段永遠會退回標準 logging。同一個檔案裡 `DataCleaningPipeline.convert_units` 也定義了兩次，實際生效的是後面那個。
- 沒有套件需求檔、沒有測試，也沒有資料；自行量測的資料沒有公開。

## 與碩論的關係

我的碩論（國立中央大學，2026）在沒有目標領域標籤的情況下，估計電池組在不同溫度下的 SOH：以 25 °C 資料訓練 CNN-LSTM 模型，再用 AdaBN 遷移到 5 °C、45 °C 及公開的 NASA PCoE 電芯。AdaBN 只用沒有標籤的目標資料重新估計 BatchNorm 的統計量，不需要反向傳播。舉例來說，5 °C 的 #8 電池組 RMSE 從 15.64% 降到 5.58%，NASA B0006 電芯從 11.68% 降到 5.01%。

這些結果來自之後的那份程式碼，無法用本 repo 的程式重現。

## 使用技術

Python · TensorFlow/Keras · NumPy · pandas · SciPy · scikit-learn · Ray Tune · SHAP · PyBaMM（選用）· Matplotlib/Plotly

## 專案結構

```text
battery_project/
├── config/            # 基本設定、硬體偵測（system_config.py）、Ray Tune 搜尋空間
├── core/              # logging 與記憶體管理
├── data/              # 清理、前處理、特徵工程、資料載入器、PyBaMM 介面
├── models/            # CNN-LSTM 基準模型、RCGAN、PINN，以及共用層（注意力、RC 參數）
├── scripts/           # 執行入口：clean_data.py、preprocess_data.py、train.py、evaluate.py
├── trainers/          # 基礎、GAN、PINN 與多溫度訓練器；損失函數；callbacks；分散式訓練
└── utils/             # 效能測試、檔案系統、環境與視覺化工具
```
