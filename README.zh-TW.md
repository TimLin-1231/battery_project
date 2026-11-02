# 以無監督遷移學習預測電池健康狀態（SOH）

[English](README.md) ｜ 繁體中文

我的碩士論文專案（國立中央大學，2026 年 6 月）：以深度學習結合無監督遷移學習，預測鋰離子電池在不同操作溫度下的健康狀態（State of Health，SOH）。

## 概述

在某個溫度下訓練的模型，拿到另一個溫度使用時，SOH 的估計準確度會下降，因為充放電特性會隨溫度改變（領域偏移，domain shift）。本專案以 25°C 的充放電資料訓練 CNN+LSTM 基準模型，再用 **AdaBN（自適應批次正規化，Adaptive Batch Normalization）** 把模型調整到新的溫度領域。AdaBN 是一種無監督遷移學習方法，不需要目標領域的標籤，也不需要反向傳播。整個流程包含資料前處理、使用可設定訓練器的模型訓練，以及評估腳本。

## 方法

- **基準模型：** 以電池充放電序列為輸入、預測 SOH 的 CNN+LSTM 網路，使用 25°C 資料訓練。
- **無監督遷移學習（AdaBN）：** 讓批次正規化的統計量對齊目標領域，把 25°C 的基準模型調整到 5°C（及其他溫度）的資料上；只使用沒有標籤的目標資料，也不需要用反向傳播重新訓練。
- **資料：** 自行量測的 25°C、5°C、45°C 充放電資料，並以公開的 NASA 電池資料集做跨資料集驗證。

## 結果

| 情境 | 自有資料 | NASA B0006 |
| --- | --- | --- |
| 25°C 基準模型直接用於 5°C 資料（領域偏移） | RMSE 15.64% | RMSE 11.68% |
| 經 AdaBN 無監督遷移後（不需標籤、不需反向傳播） | RMSE **5.58%** | RMSE **5.01%** |

## 使用技術

- Python
- Keras（深度學習模型）
- Scikit-learn（資料前處理與評估指標）

## 專案結構

```text
battery_project/
├── config/            # 基本、系統與超參數搜尋空間設定（base_config.py、system_config.py、hpo_search_space.py）
├── core/              # 日誌與記憶體工具（logging.py、memory.py）
├── data/              # BatteryDataProvider：資料集載入與分批
├── models/            # 模型定義與模型註冊表
├── scripts/           # 可直接執行的進入點：preprocess_data.py、clean_data.py、train.py、evaluate.py
├── trainers/          # 訓練邏輯：base_trainer.py、multitemp_trainer.py、gan_trainer.py、
│                      # pinn_trainer.py、losses.py、callbacks.py、distributed.py
├── utils/             # 共用輔助工具
└── logs/              # 訓練日誌與輸出（執行時產生，不納入版本控制）
```

## 執行方式

1. 安裝 Python 套件：

   ```bash
   pip install tensorflow keras scikit-learn
   ```

2. 前處理原始資料：

   ```bash
   python scripts/preprocess_data.py
   ```

3. 訓練基準模型（預設使用 `config/base_config.py`）：

   ```bash
   python scripts/train.py --config config/base_config.py
   ```

   常用選項：`--data-dir`、`--output-dir`、`--experiment-name`、`--model-name`、`--trainer-name`，以及用 Ray Tune 做超參數搜尋的 `--tune`。

4. 評估訓練好的模型：

   ```bash
   python scripts/evaluate.py
   ```

## 備註

- 自有資料集不在此公開；這套流程也支援公開的 NASA 電池資料集格式。
