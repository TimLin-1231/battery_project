#scripts/preprocess_data.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
電池數據預處理腳本 - 電池老化預測系統
提供電池數據的預處理、標準化與TFRecord轉換功能。
使用管道模式和並行處理提高效率。
新增批次處理功能，不再需要依賴批次檔案。
"""

import os
import sys
import json
import argparse
import traceback
import time
import math
import datetime
import glob
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, TypeVar
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import shutil
import concurrent.futures
import logging

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# 導入項目模組
try:
    from config.base_config import config
    from core.logging import setup_logger, LoggingTimer, log_memory_usage
    from core.memory import memory_manager
except ImportError:
    print("無法導入必要模組，使用內部配置")
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def get(self, path, default=None):
            keys = path.split('.')
            value = self
            for key in keys:
                if hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return default
            # 確保返回的不是 Config 對象，而是實際值
            return value if not isinstance(value, Config) else default
    
    # 創建默認配置
    config = Config(
        system=Config(
            data_dir="data",
            output_dir="output",
            tfrecord_dir="tfrecords",
            cache_dir="cache"
        ),
        training=Config(
            sequence_length=60
        ),
        data=Config(
            sequence_overlap=30,
            features=["time", "voltage", "current", "temp", "soc", "max_v", "min_v"],
            targets=["fdcr", "rsoc"],
            temp_values=["5deg", "25deg", "45deg"],
            test_size=0.15,
            val_size=0.15,
            random_seed=42,
            compress_zlib=False,
            chunk_size=50000,
            prefix="source"
        )
    )
    
    # 創建簡單的日誌和記憶體管理
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def setup_logger(name):
        return logging.getLogger(name)
    
    class LoggingTimer:
        def __init__(self, logger, name):
            self.logger = logger
            self.name = name
            self.start_time = None
        
        def start(self):
            self.start_time = time.time()
            self.logger.info(f"{self.name} 開始")
        
        def stop(self):
            elapsed = time.time() - self.start_time
            self.logger.info(f"{self.name} 結束，耗時 {elapsed:.2f} 秒")
    
    def log_memory_usage(logger=None, prefix=""):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            message = f"{prefix} - 記憶體使用: {memory_mb:.2f} MB"
            if logger:
                logger.info(message)
            else:
                print(message)
        except:
            pass
    
    class MemoryManager:
        def __init__(self):
            pass
        
        def collect_garbage(self):
            import gc
            gc.collect()
        
        def log_memory_usage(self, prefix=""):
            log_memory_usage(prefix=prefix)
    
    memory_manager = MemoryManager()

# 設置日誌
logger = setup_logger("preprocess")

# 類型變量
T = TypeVar('T')
PathLike = Union[str, Path]

def log_step(step_name: str):
    """記錄數據處理步驟的裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"執行 {step_name}...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{step_name} 完成，耗時 {elapsed:.2f} 秒")
                return result
            except Exception as e:
                logger.error(f"{step_name} 失敗: {e}")
                raise
        return wrapper
    return decorator

def cached_operation(cache_key_func: Callable[..., str]):
    """快取計算結果的裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 生成快取鍵
            cache_key = cache_key_func(self, *args, **kwargs)
            cache_dir = Path(self.config['cache_dir'])
            cache_path = cache_dir / f"{cache_key}.pkl"
            
            # 檢查快取是否存在
            if cache_path.exists():
                try:
                    import pickle
                    with open(cache_path, 'rb') as f:
                        result = pickle.load(f)
                    logger.info(f"從快取載入: {cache_path}")
                    return result
                except Exception as e:
                    logger.warning(f"從快取載入失敗，重新計算: {e}")
            
            # 執行原始函數
            result = func(self, *args, **kwargs)
            
            # 保存到快取
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                import pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                logger.info(f"保存到快取: {cache_path}")
            except Exception as e:
                logger.warning(f"保存到快取失敗: {e}")
            
            return result
        return wrapper
    return decorator

def get_optimal_workers():
    """獲取最佳工作線程數量"""
    try:
        cpu_count = os.cpu_count() or 4
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # 基於CPU核心數和可用記憶體決定工作線程數
        # 每個工作線程假設最多使用1GB記憶體
        workers_by_memory = max(1, int(memory_gb / 2))
        workers = min(cpu_count, workers_by_memory)
        
        # 如果是高性能機器，則增加線程數
        if cpu_count >= 8 and memory_gb >= 16:
            workers = min(cpu_count, int(memory_gb / 1.5))
        
        return max(1, min(16, workers))
    except:
        return 4  # 預設值

class DataPreprocessor:
    """電池數據預處理器，處理原始數據並轉換為TFRecord格式"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """初始化預處理器"""
        # 默認配置
        self.config = {
            'data_dir': config.get("system.data_dir", "data"),
            'output_dir': config.get("system.output_dir", "output"),
            'tfrecord_dir': config.get("system.tfrecord_dir", "tfrecords"),
            'cache_dir': config.get("system.cache_dir", "cache"),
            'sequence_length': config.get("training.sequence_length", 60),
            'overlap': config.get("data.sequence_overlap", 30),
            'feature_columns': config.get("data.features", ["time", "voltage", "current", "temp", "soc", "max_v", "min_v"]),
            'target_columns': config.get("data.targets", ["fdcr", "rsoc"]),
            'temp_values': config.get("data.temp_values", ["5deg", "25deg", "45deg"]),
            'test_size': config.get("data.test_size", 0.15),
            'val_size': config.get("data.val_size", 0.15),
            'random_seed': config.get("data.random_seed", 42),
            'compress_zlib': config.get("data.compress_zlib", False),
            'chunk_size': config.get("data.chunk_size", 50000),
            'num_parallel': get_optimal_workers(),
            'data_prefix': config.get("data.prefix", "source"),
            'buffer_size': 8 * 1024 * 1024,
            'column_map': {},
            'error_tolerance': 0.01,  # 錯誤容忍率
            'max_retries': 3,         # 最大重試次數
            'use_float16': False,     # 是否使用半精度浮點數以節省記憶體
            'safe_mode': True,         # 安全模式（更嚴格的錯誤檢查）
            'filter_charging': False,  # 是否過濾充電數據
            'charge_threshold': 0     # 充放電閾值
        }
        
        # 標準列名映射，增加更多可能的命名方式
        self.column_mapping = {
            'time': ['time', 'timestamp', 'datetime', 'date_time', 'dt', 'timeindex', 'time_index', 'DateTime'],
            'voltage': ['voltage', 'volt', 'bms_packvoltage', 'packvoltage', 'BMS_PackVoltage', 'v', 'pack_v'],
            'current': ['current', 'curr', 'bms_packcurrent', 'bms_avgcurrent', 'packcurrent', 'avgcurrent', 'BMS_PackCurrent', 'BMS_AvgCurrent', 'i'],
            'temp': ['temp', 'temperature', 'bms_temp1', 'temp1', 'celltemp', 'BMS_Temp1', 't', 'cell_temp'],
            'soc': ['soc', 'stateofcharge', 'bms_rsoc', 'bms_asoc', 'rsoc', 'asoc', 'BMS_RSOC', 'BMS_ASOC', 'charge'],
            'max_v': ['max_v', 'maxvolt', 'max_volt', 'maximum_voltage', 'maxcellvolt', 'max_cell_v', 'max_cellvolt'],
            'min_v': ['min_v', 'minvolt', 'min_volt', 'minimum_voltage', 'mincellvolt', 'min_cell_v', 'min_cellvolt'],
            'fdcr': ['fdcr', 'bms_fdcr', 'dcr_factor', 'BMS_FDCR', 'dc_resistance'],
            'rsoc': ['rsoc', 'bms_rsoc', 'relative_soc', 'relative_state_of_charge', 'BMS_RSOC', 'relativeSOC'],
            'cyclecount': ['cyclecount', 'bms_cyclecount', 'cycle_count', 'cycle', 'cycles', 'BMS_CycleCount'],
            'soh': ['soh', 'stateofhealth', 'bms_stateofhealth', 'BMS_StateOfHealth', 'health']
        }
        
        # 覆蓋配置
        if config_override:
            self.config.update(config_override)
        
        # 配置驗證
        self._validate_config()
        
        # 確保目錄存在
        self._ensure_directories()
        
        # 初始化隨機種子
        np.random.seed(self.config['random_seed'])
        tf.random.set_seed(self.config['random_seed'])
        
        # 創建線程池
        self.executor = ThreadPoolExecutor(max_workers=self.config['num_parallel'])
        
        # 紀錄初始系統狀態
        log_memory_usage(logger, "預處理器初始化")
        logger.info("電池數據預處理器初始化完成")
        logger.info(f"特徵列: {self.config['feature_columns']}")
        logger.info(f"目標列: {self.config['target_columns']}")
        logger.info(f"並行處理線程數: {self.config['num_parallel']}")
        logger.info(f"序列長度: {self.config['sequence_length']}, 重疊: {self.config['overlap']}")
    
    def _validate_config(self):
        """驗證配置參數有效性"""
        # 檢查序列長度
        if self.config['sequence_length'] <= 0:
            logger.warning(f"無效的序列長度 {self.config['sequence_length']}，調整為預設值 60")
            self.config['sequence_length'] = 60
        
        # 檢查重疊長度
        if self.config['overlap'] < 0 or self.config['overlap'] >= self.config['sequence_length']:
            logger.warning(f"無效的重疊長度 {self.config['overlap']}，調整為預設值 {self.config['sequence_length'] // 2}")
            self.config['overlap'] = self.config['sequence_length'] // 2
        
        # 檢查分割比例
        if not (0 <= self.config['test_size'] < 1):
            logger.warning(f"無效的測試集比例 {self.config['test_size']}，調整為預設值 0.15")
            self.config['test_size'] = 0.15
        
        if not (0 <= self.config['val_size'] < 1):
            logger.warning(f"無效的驗證集比例 {self.config['val_size']}，調整為預設值 0.15")
            self.config['val_size'] = 0.15
        
        # 檢查並行處理線程數
        if self.config['num_parallel'] <= 0:
            self.config['num_parallel'] = get_optimal_workers()
            logger.info(f"調整並行處理線程數為: {self.config['num_parallel']}")

    def _ensure_directories(self):
        """確保所有必要的目錄存在"""
        for dir_key in ['data_dir', 'output_dir', 'tfrecord_dir', 'cache_dir']:
            Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)
    
    def _map_columns(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """映射和派生列
        
        Returns:
            Tuple[pd.DataFrame, List[str]]: 處理後的數據框和仍然缺失的列名列表
        """
        # 記錄列映射
        mapped_columns = {}
        
        # 1. 應用用戶自定義映射
        if self.config['column_map']:
            for std_col, orig_col in self.config['column_map'].items():
                if orig_col.lower() == 'auto':
                    continue
                if orig_col in data.columns:
                    logger.info(f"用戶映射: '{orig_col}' → '{std_col}'")
                    data[std_col] = data[orig_col]
                    mapped_columns[std_col] = orig_col
        
        # 2. 自動匹配列名
        required_columns = self.config['feature_columns'] + self.config['target_columns']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            # 通過映射名找到匹配的列
            for std_col in missing_columns[:]:
                if std_col in self.column_mapping:
                    possible_names = self.column_mapping[std_col]
                    # 嘗試精確匹配
                    found_exact = False
                    for col in data.columns:
                        if col.lower() in (name.lower() for name in possible_names):
                            data[std_col] = data[col]
                            logger.info(f"自動精確映射: '{col}' → '{std_col}'")
                            missing_columns.remove(std_col)
                            mapped_columns[std_col] = col
                            found_exact = True
                            break
                    
                    # 嘗試模糊匹配
                    if not found_exact:
                        for col in data.columns:
                            if any(name.lower() in col.lower() for name in possible_names):
                                data[std_col] = data[col]
                                logger.info(f"自動模糊映射: '{col}' → '{std_col}'")
                                missing_columns.remove(std_col)
                                mapped_columns[std_col] = col
                                break
        
        # 3. 派生缺失列
        if missing_columns:
            logger.warning(f"嘗試派生缺失列: {missing_columns}")
            
            # 常見派生規則
            derivation_rules = {
                'time': lambda df: pd.to_datetime(df['DateTime']).astype('int64') // 10**9 if 'DateTime' in df.columns else None,
                'max_v': lambda df: df[[c for c in df.columns if 'cellvolt' in c.lower()]].max(axis=1) if any('cellvolt' in c.lower() for c in df.columns) else None,
                'min_v': lambda df: df[[c for c in df.columns if 'cellvolt' in c.lower()]].min(axis=1) if any('cellvolt' in c.lower() for c in df.columns) else None,
                'rsoc': lambda df: df['soc'] if 'soc' in df.columns else (df['BMS_RSOC'] if 'BMS_RSOC' in df.columns else None),
                'soc': lambda df: df['rsoc'] if 'rsoc' in df.columns else (df['BMS_RSOC'] if 'BMS_RSOC' in df.columns else None),
                'voltage': lambda df: df[[c for c in df.columns if 'cellvolt' in c.lower()]].mean(axis=1) if any('cellvolt' in c.lower() for c in df.columns) else None,
                'temp': lambda df: df['BMS_Temp1'] if 'BMS_Temp1' in df.columns else None,
                'current': lambda df: df['BMS_PackCurrent'] if 'BMS_PackCurrent' in df.columns else None
            }
            
            # 應用派生規則
            for col in missing_columns[:]:
                # 檢查是否有規則
                if col in derivation_rules:
                    derived_value = derivation_rules[col](data)
                    if derived_value is not None:
                        data[col] = derived_value
                        logger.info(f"派生列: '{col}'")
                        missing_columns.remove(col)
                        mapped_columns[col] = f"派生自規則"
                        continue
                
                # 嘗試通過前綴/後綴查找
                matching_cols = [c for c in data.columns if col.lower() in c.lower()]
                if matching_cols:
                    data[col] = data[matching_cols[0]]
                    logger.info(f"從'{matching_cols[0]}'派生'{col}'")
                    missing_columns.remove(col)
                    mapped_columns[col] = matching_cols[0]
        
        # 記錄列映射結果
        logger.info(f"列映射結果: {mapped_columns}")
        if missing_columns:
            logger.warning(f"仍然缺少列: {missing_columns}")
        
        return data, missing_columns
    
    def _find_cell_voltage_columns(self, data: pd.DataFrame) -> List[str]:
        """查找數據中的電池單元電壓列"""
        cell_volt_cols = []
        patterns = ['cellvolt', 'cell_volt', 'CellVolt']
        
        for col in data.columns:
            if any(pattern in col for pattern in patterns):
                # 確認其值範圍符合電池單元電壓
                try:
                    if data[col].min() >= 2.0 and data[col].max() <= 4.5:
                        cell_volt_cols.append(col)
                except:
                    pass
        
        return cell_volt_cols
    
    def _check_and_fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """檢查並修正數據類型，確保為浮點數"""
        # 確保所有目標列都存在
        for col in self.config['feature_columns'] + self.config['target_columns']:
            if col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    # 如果啟用了半精度浮點數以節省記憶體
                    if self.config['use_float16']:
                        data[col] = data[col].astype(np.float16)
                    else:
                        data[col] = data[col].astype(np.float32)
                except Exception as e:
                    logger.warning(f"轉換列 '{col}' 到數值類型時出錯: {e}")
        
        return data
    
    @log_step("加載並清理數據")
    def load_and_clean_data(self, filepath: PathLike) -> pd.DataFrame:
        """加載並清理原始數據 (優化版)"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"檔案不存在: {filepath}")
            
        retry_count = 0
        while retry_count < self.config['max_retries']:
            try:
                # 根據文件格式加載數據
                if filepath.suffix.lower() == '.csv':
                    try:
                        data = pd.read_csv(filepath, low_memory=False)
                    except (pd.errors.ParserError, UnicodeDecodeError) as e:
                        logger.warning(f"標準方式讀取CSV失敗: {e}，嘗試其他編碼")
                        # 嘗試不同編碼
                        for encoding in ['utf-8', 'latin1', 'cp1252', 'utf-8-sig']:
                            try:
                                data = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                                logger.info(f"使用編碼 {encoding} 成功讀取CSV")
                                break
                            except (pd.errors.ParserError, UnicodeDecodeError):
                                continue
                        else:
                            raise IOError(f"無法以任何編碼方式讀取CSV文件: {filepath}")
                
                elif filepath.suffix.lower() == '.parquet':
                    data = pd.read_parquet(filepath)
                else:
                    raise ValueError(f"不支持的文件格式: {filepath.suffix}")
                
                # 檢查數據有效性
                if data.empty:
                    raise ValueError(f"載入的數據為空: {filepath}")
                
                logger.info(f"載入數據: {filepath}, 形狀: {data.shape}")
                
                # 列名映射和派生
                data, missing_columns = self._map_columns(data)
                
                # 檢查是否仍有缺失列
                if missing_columns:
                    tolerance_threshold = len(self.config['feature_columns'] + self.config['target_columns']) * self.config['error_tolerance']
                    if self.config['safe_mode'] and len(missing_columns) > tolerance_threshold:
                        logger.error(f"數據仍缺少過多必要列: {missing_columns}")
                        logger.error(f"可用列: {list(data.columns)}")
                        raise ValueError(f"數據缺少必要列: {missing_columns}")
                    else:
                        logger.warning(f"數據缺少一些列但在容忍範圍內，將嘗試繼續: {missing_columns}")
                
                # 檢查數據類型
                data = self._check_and_fix_data_types(data)
                
                # 處理缺失值
                na_before = data.isna().sum().sum()
                if na_before > 0:
                    logger.warning(f"數據中發現 {na_before} 個缺失值，進行補全")
                    # 先使用前後值填充
                    data = data.fillna(method='ffill').fillna(method='bfill')
                    
                    # 檢查是否還有缺失值
                    na_after = data.isna().sum().sum()
                    if na_after > 0:
                        logger.warning(f"填充後仍有 {na_after} 個缺失值，使用列平均值填充")
                        # 使用平均值填充剩餘缺失值
                        data = data.fillna(data.mean())
                
                # 處理異常值 (使用分位數進行魯棒處理)
                for col in self.config['feature_columns'] + self.config['target_columns']:
                    if col in data.columns and data[col].dtype.kind in 'fc':
                        # 使用分位數而非均值標準差以提高魯棒性
                        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound, upper_bound = q1 - 3 * iqr, q3 + 3 * iqr
                        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                        
                        if outliers > 0:
                            logger.warning(f"列 '{col}' 中發現 {outliers} 個異常值，進行修正")
                            data.loc[data[col] < lower_bound, col] = lower_bound
                            data.loc[data[col] > upper_bound, col] = upper_bound
                
                break  # 如果成功載入和處理，跳出重試循環
            
            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, IOError) as e:
                # 處理預期中的異常
                retry_count += 1
                logger.error(f"加載數據時出錯 (嘗試 {retry_count}/{self.config['max_retries']}): {type(e).__name__}: {e}")
                if retry_count >= self.config['max_retries']:
                    raise RuntimeError(f"無法載入數據，已達最大重試次數: {e}")
                time.sleep(1)  # 間隔1秒後重試
            
            except Exception as e:
                # 處理未預期的異常
                logger.error(f"載入數據時發生未預期錯誤: {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                raise  # 未預期的錯誤直接重新拋出

            # 處理充放電數據
        if self.config.get('process_charge_discharge', True):
            # 標記充放電
            data = self.mark_charge_discharge(data, current_col='current', 
                                            threshold=self.config.get('charge_threshold', 0))
            
            # 如果配置為過濾充電數據
            if self.config.get('filter_charging', False):
                data = self.filter_charging_data(data, keep_charging=False)
        
        # 記錄記憶體使用
        log_memory_usage(logger, "數據載入後")
        return data
    
    @log_step("按溫度分割數據")
    def split_by_temperature(self, data: pd.DataFrame, temp_column: str = 'temp') -> Dict[str, pd.DataFrame]:
        """按溫度分割數據 (使用實際溫度範圍作為標籤)"""
        # 檢查溫度列是否存在
        if temp_column not in data.columns:
            logger.warning(f"溫度列 '{temp_column}' 不存在，嘗試尋找替代列")
            
            # 嘗試尋找替代的溫度列
            temp_aliases = ['temp', 'temperature', 'bms_temp1', 'BMS_Temp1', 't', 'cell_temp']
            for alias in temp_aliases:
                if alias in data.columns:
                    temp_column = alias
                    logger.info(f"使用替代溫度列: {temp_column}")
                    break
            else:
                logger.warning(f"找不到有效的溫度列，跳過溫度分割")
                return {"combined": data}
        
        # 取溫度的不同值
        try:
            # 先移除缺失值和異常值
            valid_temps = data[temp_column].dropna()
            if valid_temps.empty:
                logger.warning("溫度列中沒有有效值，跳過溫度分割")
                return {"combined": data}
                
            # 使用IQR方法移除極端異常溫度
            q1, q3 = valid_temps.quantile(0.25), valid_temps.quantile(0.75)
            iqr = q3 - q1
            valid_temps = valid_temps[(valid_temps >= q1 - 3*iqr) & (valid_temps <= q3 + 3*iqr)]
            
            temp_values = valid_temps.unique()
            min_temp, max_temp = valid_temps.min(), valid_temps.max()
            logger.info(f"數據中發現 {len(temp_values)} 個不同溫度值，範圍從 {min_temp:.2f} 到 {max_temp:.2f}")
            
            # 根據溫度值的分佈情況決定分割策略
            temp_dfs = {}
            
            # 默認分3組，但如果溫度值很少，則減少組數
            num_groups = min(3, len(temp_values))
            
            if len(temp_values) <= num_groups:
                # 少量離散溫度值：每個溫度一組
                sorted_temps = sorted(temp_values)
                for i, temp in enumerate(sorted_temps):
                    # 使用實際溫度作為標籤
                    label = f"{temp:.1f}C"
                    temp_df = data[data[temp_column] == temp].copy()
                    temp_dfs[label] = temp_df
                    logger.info(f"溫度 {temp:.2f} → 標籤 {label}: {len(temp_df)} 行")
            else:
                # 使用分位數確定分割點，確保每組數據量均衡
                if num_groups == 1:
                    percentiles = []
                elif num_groups == 2:
                    percentiles = [0.5]
                else:  # num_groups == 3
                    percentiles = [0.33, 0.67]
                    
                # 計算分位數作為分割點
                split_points = [min_temp] + [float(valid_temps.quantile(p)) for p in percentiles] + [max_temp]
                split_points = sorted(list(set(split_points)))  # 確保不重複
                
                logger.info(f"溫度分割點: {[f'{t:.2f}' for t in split_points]}")
                
                # 根據分割點分組
                for i in range(len(split_points)-1):
                    lower = split_points[i]
                    upper = split_points[i+1]
                    label = f"{lower:.1f}-{upper:.1f}C"
                    
                    if i == len(split_points)-2:  # 最後一組
                        temp_df = data[(data[temp_column] >= lower) & (data[temp_column] <= upper)].copy()
                    else:
                        temp_df = data[(data[temp_column] >= lower) & (data[temp_column] < upper)].copy()
                    
                    if not temp_df.empty:
                        temp_dfs[label] = temp_df
                        logger.info(f"溫度範圍 {lower:.2f}-{upper:.2f} → 標籤 {label}: {len(temp_df)} 行")
            
            # 處理未分類數據
            if temp_dfs:
                all_assigned = pd.concat(temp_dfs.values())
                unassigned = data.loc[~data.index.isin(all_assigned.index)]
                if not unassigned.empty:
                    logger.warning(f"有 {len(unassigned)} 行數據未分配到任何溫度標籤")
                    label = f"other_{min_temp:.1f}-{max_temp:.1f}C"
                    temp_dfs[label] = unassigned
            else:
                # 如果都沒分到組，返回完整數據
                label = f"{min_temp:.1f}-{max_temp:.1f}C"
                temp_dfs[label] = data
                logger.info(f"所有數據 → 標籤 {label}: {len(data)} 行")
                
            return temp_dfs
                
        except Exception as e:
            logger.error(f"按溫度分割數據時出錯: {e}")
            logger.error(traceback.format_exc())
            logger.warning("返回未分割的完整數據集")
            return {"combined": data}
    
    def _create_sequence_chunk(self, data_chunk: np.ndarray, seq_length: int, step: int) -> List[np.ndarray]:
        """從數據塊創建序列窗口"""
        sequences = []
        n_sequences = (len(data_chunk) - seq_length) // step + 1
        
        for i in range(n_sequences):
            start_idx = i * step
            end_idx = start_idx + seq_length
            if end_idx <= len(data_chunk):
                sequences.append(data_chunk[start_idx:end_idx])
        
        return sequences
    
    @log_step("創建序列窗口")
    def create_sequences(self, data: pd.DataFrame, timeseries_column: str = 'time') -> Tuple[np.ndarray, np.ndarray]:
        """將時間序列數據轉換為序列窗口
        
        改進版本：使用預分配陣列和生成器減少記憶體使用
        """
        seq_length = self.config['sequence_length']
        overlap = self.config['overlap']
        step = seq_length - overlap
        
        # 檢查序列長度和數據長度
        if seq_length <= 0:
            raise ValueError(f"序列長度必須為正數")
        
        if len(data) < seq_length:
            logger.warning(f"數據長度 ({len(data)}) 小於序列長度 ({seq_length})，嘗試調整...")
            
            if len(data) < 10:  # 數據太少，無法處理
                raise ValueError(f"數據行數過少 ({len(data)})，無法創建序列")
            
            # 自動調整序列長度為數據長度的一半
            new_seq_length = max(10, len(data) // 2)  # 保證最小長度
            logger.warning(f"自動調整序列長度為 {new_seq_length}")
            seq_length = new_seq_length
            overlap = seq_length // 2
            step = seq_length - overlap
        
        # 提取特徵和目標列
        features_cols = sorted([col for col in self.config['feature_columns'] if col in data.columns])
        targets_cols = sorted([col for col in self.config['target_columns'] if col in data.columns])
        
        if not features_cols:
            raise ValueError(f"找不到任何特徵列: {self.config['feature_columns']}")
        
        if not targets_cols:
            raise ValueError(f"找不到任何目標列: {self.config['target_columns']}")
        
        # 提取數值
        features = data[features_cols].values
        targets = data[targets_cols].values
        
        # 計算序列數量
        n_sequences = (len(data) - seq_length) // step + 1
        logger.info(f"將創建 {n_sequences} 個序列，步長 {step}，重疊 {overlap}")
        
        if n_sequences <= 0:
            raise ValueError(f"無法創建序列：數據長度 {len(data)}，序列長度 {seq_length}，步長 {step}")
        
        # 使用生成器函數減少中間列表的記憶體占用
        def sequence_generator():
            for i in range(n_sequences):
                start_idx = i * step
                end_idx = start_idx + seq_length
                yield features[start_idx:end_idx], targets[start_idx:end_idx]
        
        # 預分配陣列，避免動態擴展
        X_sequences = np.zeros((n_sequences, seq_length, len(features_cols)), dtype=np.float32)
        y_sequences = np.zeros((n_sequences, seq_length, len(targets_cols)), dtype=np.float32)
        
        # 填充序列
        with tqdm(total=n_sequences, desc="創建序列") as pbar:
            for i, (X_seq, y_seq) in enumerate(sequence_generator()):
                X_sequences[i] = X_seq
                y_sequences[i] = y_seq
                pbar.update(1)
        
        # 記錄特徵列和目標列的名稱
        self.feature_names = features_cols
        self.target_names = targets_cols
        
        # 記錄序列形狀
        logger.info(f"最終序列形狀: X:{X_sequences.shape}, y:{y_sequences.shape}")
        log_memory_usage(logger, "創建序列後")
        
        return X_sequences, y_sequences
    
    @log_step("計算標準化參數")
    def compute_scaler_params(self, X_sequences: np.ndarray, y_sequences: np.ndarray) -> Dict[str, float]:
        """計算標準化參數"""
        scaler_params = {}
        
        # 檢查輸入數據
        if X_sequences.size == 0 or y_sequences.size == 0:
            logger.error("無法計算標準化參數：輸入序列為空")
            raise ValueError("輸入序列為空")
        
        # 特徵標準化參數
        for i, col in enumerate(self.feature_names):
            feature_data = X_sequences[:, :, i].flatten()
            # 使用更魯棒的統計方法
            # 移除極端異常值
            q1, q3 = np.percentile(feature_data, [25, 75])
            iqr = q3 - q1
            valid_data = feature_data[(feature_data >= q1 - 3*iqr) & (feature_data <= q3 + 3*iqr)]
            
            if len(valid_data) > 0:
                scaler_params[f"X_{col}_mean"] = float(np.mean(valid_data))
                scaler_params[f"X_{col}_scale"] = float(np.std(valid_data) or 1.0)
                # 記錄其他統計信息
                scaler_params[f"X_{col}_min"] = float(np.min(valid_data))
                scaler_params[f"X_{col}_max"] = float(np.max(valid_data))
            else:
                logger.warning(f"特徵 {col} 沒有有效數據用於計算標準化參數")
                scaler_params[f"X_{col}_mean"] = 0.0
                scaler_params[f"X_{col}_scale"] = 1.0
        
        # 目標標準化參數
        for i, col in enumerate(self.target_names):
            target_data = y_sequences[:, :, i].flatten()
            # 同樣移除極端異常值
            q1, q3 = np.percentile(target_data, [25, 75])
            iqr = q3 - q1
            valid_data = target_data[(target_data >= q1 - 3*iqr) & (target_data <= q3 + 3*iqr)]
            
            if len(valid_data) > 0:
                scaler_params[f"y_{col}_mean"] = float(np.mean(valid_data))
                scaler_params[f"y_{col}_scale"] = float(np.std(valid_data) or 1.0)
                # 記錄其他統計信息
                scaler_params[f"y_{col}_min"] = float(np.min(valid_data))
                scaler_params[f"y_{col}_max"] = float(np.max(valid_data))
            else:
                logger.warning(f"目標 {col} 沒有有效數據用於計算標準化參數")
                scaler_params[f"y_{col}_mean"] = 0.0
                scaler_params[f"y_{col}_scale"] = 1.0
        
        # 記錄序列形狀和特徵/目標名稱
        scaler_params["Xs_shape"] = str(list(X_sequences.shape))
        scaler_params["Ys_shape"] = str(list(y_sequences.shape))
        scaler_params["feature_names"] = str(self.feature_names)
        scaler_params["target_names"] = str(self.target_names)
        
        # 記錄處理日期和版本信息
        scaler_params["processing_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        scaler_params["version"] = "2.0.0"
        
        return scaler_params
    
    @log_step("保存標準化參數")
    def save_scaler_params(self, scaler_params: Dict[str, float], output_path: PathLike) -> None:
        """保存標準化參數"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存為Parquet和JSON (雙份以相容不同載入方式)
        parquet_path = output_path
        json_path = output_path.with_suffix('.json')
        
        try:
            # 轉換為DataFrame並保存
            scaler_df = pd.DataFrame({k: [v] for k, v in scaler_params.items()})
            scaler_df.to_parquet(parquet_path, index=False)
            
            # 保存JSON格式
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(scaler_params, f, indent=2, ensure_ascii=False)
            
            logger.info(f"標準化參數已保存為 {parquet_path} 和 {json_path}")
            
            # 驗證是否可以正確讀取
            try:
                # 驗證Parquet
                test_parquet = pd.read_parquet(parquet_path)
                # 驗證JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    test_json = json.load(f)
                
                logger.info("驗證標準化參數讀取成功")
            except Exception as e:
                logger.warning(f"驗證標準化參數時出錯: {e}")
        
        except Exception as e:
            logger.error(f"保存標準化參數時出錯: {e}")
            # 嘗試僅保存JSON
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(scaler_params, f, indent=2, ensure_ascii=False)
                logger.info(f"僅保存了JSON格式: {json_path}")
            except Exception as e2:
                logger.error(f"保存JSON標準化參數也失敗: {e2}")
    
    @log_step("應用標準化")
    def apply_scaling(self, X_sequences: np.ndarray, y_sequences: np.ndarray, 
                      scaler_params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """應用標準化"""
        # 創建標準化數據的副本
        X_scaled = X_sequences.copy()
        y_scaled = y_sequences.copy()
        
        # 檢查特徵和目標名稱
        feature_names = self.feature_names
        target_names = self.target_names
        
        # 如果參數中有保存的名稱，從參數中讀取
        if "feature_names" in scaler_params:
            try:
                stored_features = eval(scaler_params["feature_names"])
                if len(stored_features) == X_sequences.shape[2]:
                    feature_names = stored_features
                    logger.info(f"使用存儲的特徵名稱: {feature_names}")
            except:
                logger.warning("無法解析存儲的特徵名稱，使用當前特徵名稱")
        
        if "target_names" in scaler_params:
            try:
                stored_targets = eval(scaler_params["target_names"])
                if len(stored_targets) == y_sequences.shape[2]:
                    target_names = stored_targets
                    logger.info(f"使用存儲的目標名稱: {target_names}")
            except:
                logger.warning("無法解析存儲的目標名稱，使用當前目標名稱")
        
        # 並行應用特徵標準化
        def scale_feature(i):
            col = feature_names[i]
            mean_key = f"X_{col}_mean"
            scale_key = f"X_{col}_scale"
            
            if mean_key in scaler_params and scale_key in scaler_params:
                mean = scaler_params[mean_key]
                scale = scaler_params[scale_key]
                
                # 避免零除
                if scale == 0 or pd.isna(scale):
                    scale = 1.0
                
                return i, (X_sequences[:, :, i] - mean) / scale
            return i, X_sequences[:, :, i]
        
        # 並行應用目標標準化
        def scale_target(i):
            col = target_names[i]
            mean_key = f"y_{col}_mean"
            scale_key = f"y_{col}_scale"
            
            if mean_key in scaler_params and scale_key in scaler_params:
                mean = scaler_params[mean_key]
                scale = scaler_params[scale_key]
                
                # 避免零除
                if scale == 0 or pd.isna(scale):
                    scale = 1.0
                
                return i, (y_sequences[:, :, i] - mean) / scale
            return i, y_sequences[:, :, i]
        
        try:
            # 執行並行處理
            with ThreadPoolExecutor(max_workers=self.config['num_parallel']) as executor:
                # 特徵標準化
                future_features = [executor.submit(scale_feature, i) 
                                for i in range(X_sequences.shape[2])]
                
                # 目標標準化
                future_targets = [executor.submit(scale_target, i) 
                                for i in range(y_sequences.shape[2])]
                
                # 收集結果
                for future in as_completed(future_features):
                    i, scaled = future.result()
                    X_scaled[:, :, i] = scaled
                
                for future in as_completed(future_targets):
                    i, scaled = future.result()
                    y_scaled[:, :, i] = scaled
            
            # 檢查是否有NaN或無限值
            nan_count_X = np.isnan(X_scaled).sum()
            nan_count_y = np.isnan(y_scaled).sum()
            
            if nan_count_X > 0 or nan_count_y > 0:
                logger.warning(f"標準化後發現 {nan_count_X} 個NaN值在特徵中，{nan_count_y} 個NaN值在目標中")
                # 替換NaN值為0
                X_scaled = np.nan_to_num(X_scaled)
                y_scaled = np.nan_to_num(y_scaled)
                logger.info("已將NaN值替換為0")
            
            inf_count_X = np.isinf(X_scaled).sum()
            inf_count_y = np.isinf(y_scaled).sum()
            
            if inf_count_X > 0 or inf_count_y > 0:
                logger.warning(f"標準化後發現 {inf_count_X} 個無限值在特徵中，{inf_count_y} 個無限值在目標中")
                # 替換無限值為最大浮點數
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e30, neginf=-1e30)
                y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=1e30, neginf=-1e30)
                logger.info("已處理無限值")
        
        except Exception as e:
            logger.error(f"標準化過程中出錯: {e}")
            logger.warning("返回未標準化的數據")
            return X_sequences, y_sequences
        
        return X_scaled, y_scaled
    
    @log_step("分割數據集 (LOOCV)")
    def split_data(self, X_sequences: np.ndarray, y_sequences: np.ndarray, channel_ids: List[str]) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """使用留一法交叉驗證 (LOOCV) 分割數據集
        
        Args:
            X_sequences: 特徵序列數據 (n_samples, seq_length, n_features)
            y_sequences: 目標序列數據 (n_samples, seq_length, n_targets)
            channel_ids: 每個樣本對應的電池通道標識符列表 (長度與 n_samples 相同)
        
        Returns:
            List[Dict[str, Tuple[np.ndarray, np.ndarray]]]: 每個折疊的數據分割列表
        """
        if len(channel_ids) != X_sequences.shape[0]:
            raise ValueError(f"通道標識數量 ({len(channel_ids)}) 與樣本數量 ({X_sequences.shape[0]}) 不匹配")

        # 獲取唯一的電池通道
        unique_channels = sorted(set(channel_ids))  # 例如 ['Ch1', 'Ch2', 'Ch3', 'Ch4']
        n_folds = len(unique_channels)
        
        if n_folds < 2:
            logger.warning(f"僅檢測到 {n_folds} 個獨立通道，無法執行 LOOCV，返回單一訓練集")
            return [{
                'train': (X_sequences, y_sequences),
                'val': (np.array([]), np.array([])),
                'test': (np.array([]), np.array([]))
            }]
        
        logger.info(f"執行 {n_folds}-折 LOOCV，基於通道: {unique_channels}")
        
        # 將數據按通道分組
        channel_indices = {ch: [] for ch in unique_channels}
        for idx, ch in enumerate(channel_ids):
            channel_indices[ch].append(idx)
        
        # 儲存每個折疊的分割
        folds = []
        
        for test_ch in unique_channels:
            logger.info(f"折疊: 測試通道 = {test_ch}")
            
            # 測試集索引
            test_indices = np.array(channel_indices[test_ch])
            
            # 訓練集索引（除去測試通道的所有數據）
            train_indices = []
            for ch in unique_channels:
                if ch != test_ch:
                    train_indices.extend(channel_indices[ch])
            train_indices = np.array(train_indices)
            
            # 從訓練集中隨機抽取驗證集（如果需要）
            val_indices = np.array([])
            if len(train_indices) > 1 and self.config.get('val_size', 0) > 0:
                # 按比例隨機選擇驗證集
                val_size = max(1, int(len(train_indices) * self.config['val_size']))
                np.random.shuffle(train_indices)
                val_indices = train_indices[:val_size]
                train_indices = train_indices[val_size:]
            
            # 創建分割
            fold_split = {
                'train': (X_sequences[train_indices], y_sequences[train_indices]),
                'val': (X_sequences[val_indices], y_sequences[val_indices]) if len(val_indices) > 0 else (np.array([]), np.array([])),
                'test': (X_sequences[test_indices], y_sequences[test_indices])
            }
            
            # 記錄分割大小
            for split_name, (X, y) in fold_split.items():
                if X.size > 0:
                    logger.info(f"折疊 {test_ch} - {split_name} 集: {X.shape[0]} 個樣本")
                else:
                    logger.warning(f"折疊 {test_ch} - {split_name} 集為空")
            
            folds.append(fold_split)
        
        return folds
    
    @staticmethod
    def _write_tfrecord_chunk(X: np.ndarray, y: np.ndarray, output_path: Path,
                            start_idx: int, end_idx: int,
                            compress_zlib: bool) -> int:
        """寫入TFRecord數據塊 (優化版靜態工作函數)"""
        options = tf.io.TFRecordOptions(
            compression_type="ZLIB" if compress_zlib else ""
        )
        count = 0

        try:
            # 確保目錄存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用 tf.io.TFRecordWriter
            with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
                for i in range(start_idx, end_idx):
                    if i >= len(X):
                        break
                    
                    # 處理數據並寫入
                    x_sample = np.nan_to_num(X[i].flatten(), nan=0.0, 
                                            posinf=np.finfo(np.float32).max, 
                                            neginf=np.finfo(np.float32).min)
                    y_sample = np.nan_to_num(y[i].flatten(), nan=0.0, 
                                            posinf=np.finfo(np.float32).max, 
                                            neginf=np.finfo(np.float32).min)

                    feature = {
                        'Xs': tf.train.Feature(float_list=tf.train.FloatList(value=x_sample)),
                        'y': tf.train.Feature(float_list=tf.train.FloatList(value=y_sample))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    count += 1
                        
            return count
        except Exception as e:
            # 詳細記錄錯誤
            error_msg = f"寫入TFRecord塊 {output_path} 時出錯: {e}"
            print(f"[Worker Error] {error_msg}")
            
            # 清理可能損壞的檔案
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception as unlink_e:
                    print(f"[Worker Error] 刪除錯誤檔案 {output_path} 失敗: {unlink_e}")
            
            # 創建一個空的有效文件
            try:
                with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
                    empty_x = np.array([0.0], dtype=np.float32)
                    empty_y = np.array([0.0], dtype=np.float32)
                    feature = {
                        'Xs': tf.train.Feature(float_list=tf.train.FloatList(value=empty_x)),
                        'y': tf.train.Feature(float_list=tf.train.FloatList(value=empty_y))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                return 1
            except Exception:
                return 0
    
    @log_step("寫入TFRecord")
    def write_tfrecord(self, X: np.ndarray, y: np.ndarray, output_path: PathLike) -> int:
        """將數據寫入TFRecord檔案 (優化版本，包含並行合併)"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 檢查空數據情況
        if X.size == 0 or y.size == 0:
            logger.warning("輸入數據為空，創建空的TFRecord檔案")
            options = tf.io.TFRecordOptions(
                compression_type="ZLIB" if self.config['compress_zlib'] else ""
            )
            try:
                with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
                    pass
                logger.info(f"已創建空的TFRecord檔案: {output_path}")
                return 0
            except Exception as e:
                logger.error(f"創建空的TFRecord檔案失敗: {e}")
                return 0

        n_samples = X.shape[0]
        num_workers = max(1, min(self.config['num_parallel'], 8))
        samples_per_worker = max(1, math.ceil(n_samples / num_workers))

        # 創建臨時檔案路徑和分塊參數
        temp_files = []
        chunks = []
        for i in range(num_workers):
            start_idx = i * samples_per_worker
            end_idx = min(start_idx + samples_per_worker, n_samples)
            if start_idx < end_idx:
                temp_file = output_path.with_name(f"{output_path.stem}_temp_{i}{output_path.suffix}")
                temp_files.append(temp_file)
                chunks.append((X, y, temp_file, start_idx, end_idx, self.config['compress_zlib']))

        # 並行寫入
        total_count = 0
        with tqdm(total=n_samples, desc=f"寫入 {output_path.name}") as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(DataPreprocessor._write_tfrecord_chunk, *chunk) for chunk in chunks]
                for future in as_completed(futures):
                    try:
                        count = future.result(timeout=60)  # 添加超時
                        total_count += count
                        pbar.update(count)
                    except Exception as e:
                        logger.error(f"寫入塊時出錯: {e}")

        # 並行合併臨時文件
        def merge_chunk(files, final_path, compress_zlib):
            options = tf.io.TFRecordOptions(
                compression_type="ZLIB" if compress_zlib else ""
            )
            count = 0
            try:
                with tf.io.TFRecordWriter(str(final_path), options=options) as writer:
                    for file_path in files:
                        if file_path.exists() and file_path.stat().st_size > 0:
                            dataset = tf.data.TFRecordDataset(
                                str(file_path), 
                                compression_type="ZLIB" if compress_zlib else ""
                            )
                            for record in dataset:
                                writer.write(record.numpy())
                                count += 1
                            # 合併後刪除臨時檔案
                            file_path.unlink(missing_ok=True)
                return count
            except Exception as e:
                logger.error(f"合併臨時檔案時出錯: {e}")
                return 0

        # 如果只有一個臨時檔案且成功寫入，則直接重命名
        if len(temp_files) == 1 and temp_files[0].exists() and temp_files[0].stat().st_size > 0:
            try:
                temp_files[0].rename(output_path)
                logger.info(f"單一臨時檔案直接重命名為 {output_path}")
                return total_count
            except Exception as e:
                logger.error(f"重命名臨時檔案失敗: {e}")
                # 繼續正常合併流程
        
        # 多個檔案情況下的並行合併
        logger.info(f"合併 {len(temp_files)} 個臨時檔案到 {output_path}")
        
        # 確保輸出檔案不存在
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception as e:
                logger.warning(f"無法刪除現有檔案: {e}")
                output_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        
        # 對臨時檔案分組以進行並行合併
        merged_count = 0
        chunk_size = max(1, len(temp_files) // num_workers)
        file_chunks = [temp_files[i:i+chunk_size] for i in range(0, len(temp_files), chunk_size)]
        
        if len(file_chunks) == 1:
            # 單組檔案直接合併到最終輸出
            merged_count = merge_chunk(file_chunks[0], output_path, self.config['compress_zlib'])
        else:
            # 多組檔案先分別合併到中間檔案，再合併到最終輸出
            intermediate_files = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i, file_chunk in enumerate(file_chunks):
                    inter_path = output_path.with_name(f"{output_path.stem}_inter_{i}{output_path.suffix}")
                    intermediate_files.append(inter_path)
                    futures.append(executor.submit(merge_chunk, file_chunk, inter_path, self.config['compress_zlib']))
                
                # 等待所有中間合併完成
                for future in as_completed(futures):
                    merged_count += future.result()
            
            # 最終合併中間檔案
            final_count = merge_chunk(intermediate_files, output_path, self.config['compress_zlib'])
            logger.info(f"最終合併了 {final_count} 條記錄到 {output_path}")
        
        # 清理殘留的臨時檔案
        for file_path in temp_files + (intermediate_files if 'intermediate_files' in locals() else []):
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass

        logger.info(f"成功寫入並合併 {total_count} 個樣本到 {output_path}")
        return total_count
    
    @log_step("處理並轉換文件 (LOOCV)")
    def process_file(self, input_path: PathLike, output_prefix: Optional[str] = None, apply_scaling: bool = True) -> bool:
        """完整處理單個文件的管道，從加載到TFRecord輸出，使用 LOOCV (優化版)"""
        input_path = Path(input_path)
        
        # 確定輸出前綴
        if output_prefix is None:
            output_prefix = input_path.stem
        
        # 構建輸出路徑
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 步驟1：加載並清理數據
            data = self.load_and_clean_data(input_path)
            
            # 步驟2：按溫度分割數據
            temp_dfs = self.split_by_temperature(data)
            
            # 為每個溫度處理數據
            for temp_label, temp_df in temp_dfs.items():
                logger.info(f"處理溫度 '{temp_label}' 數據 ({len(temp_df)} 行)")
                
                # 更靈活的通道標識推斷
                if 'channel' not in temp_df.columns:
                    # 1. 先嘗試從檔案路徑推斷
                    channel_match = re.search(r'Ch\d', str(input_path))
                    if channel_match:
                        channel_id = channel_match.group()
                        logger.info(f"從檔案路徑推斷通道標識: {channel_id}")
                    # 2. 嘗試尋找其他可能的通道標識列
                    elif any(col.lower() in ['channel', 'ch', 'cell'] for col in temp_df.columns):
                        channel_cols = [col for col in temp_df.columns if col.lower() in ['channel', 'ch', 'cell']]
                        channel_id = str(temp_df[channel_cols[0]].iloc[0])
                        logger.info(f"從列 '{channel_cols[0]}' 獲取通道標識: {channel_id}")
                    # 3. 如果無法推斷，使用其他標識符或預設值
                    else:
                        # 使用檔案名部分作為通道標識
                        channel_id = f"Ch_{input_path.stem[:6]}"
                        logger.warning(f"無法確定通道標識，創建標識: {channel_id}")
                    
                    # 添加通道列
                    temp_df['channel'] = channel_id
                
                # 步驟3：創建序列窗口
                X_sequences, y_sequences = self.create_sequences(temp_df)
                
                # 正確提取通道標識
                # 確保通道標識與序列對齊
                seq_step = self.config['sequence_length'] - self.config['overlap']
                channel_indices = range(0, len(temp_df), seq_step)
                channel_indices = [i for i in channel_indices if i + self.config['sequence_length'] <= len(temp_df)]
                
                if len(channel_indices) >= X_sequences.shape[0]:
                    channel_ids = [temp_df['channel'].iloc[i] for i in channel_indices[:X_sequences.shape[0]]]
                else:
                    # 如果索引不足，使用最後一個值填充
                    channel_ids = [temp_df['channel'].iloc[i] for i in channel_indices]
                    if channel_ids:
                        channel_ids.extend([channel_ids[-1]] * (X_sequences.shape[0] - len(channel_ids)))
                    else:
                        # 極端情況：使用默認值
                        channel_ids = [f"Ch_default"] * X_sequences.shape[0]
                        logger.warning(f"無法提取足夠的通道標識，使用默認值填充: {channel_ids[0]}")
                
                logger.info(f"產生了 {len(channel_ids)} 個通道標識用於 {X_sequences.shape[0]} 個序列")
                
                # 步驟4：計算標準化參數
                scaler_params = self.compute_scaler_params(X_sequences, y_sequences)
                
                # 保存標準化參數
                scaler_path = output_dir / f"{output_prefix}_{temp_label}_scaler.parquet"
                self.save_scaler_params(scaler_params, scaler_path)
                
                # 步驟5：應用標準化（如果需要）
                if apply_scaling:
                    X_scaled, y_scaled = self.apply_scaling(X_sequences, y_sequences, scaler_params)
                else:
                    X_scaled, y_scaled = X_sequences, y_sequences
                
                # 步驟6：執行 LOOCV 分割
                folds = self.split_data(X_scaled, y_scaled, channel_ids)
                
                # 步驟7：為每個折疊寫入TFRecord
                for fold_idx, fold_split in enumerate(folds):
                    for split_name, (split_X, split_y) in fold_split.items():
                        if split_X.size == 0 or split_y.size == 0:
                            logger.warning(f"折疊 {fold_idx} - 跳過空的 {split_name} 分割")
                            continue
                        
                        # 設置TFRecord路徑，包含折疊編號
                        tfrecord_path = Path(self.config['tfrecord_dir']) / f"{output_prefix}_{temp_label}_fold{fold_idx}_{split_name}.tfrecord"
                        tfrecord_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 寫入TFRecord
                        count = self.write_tfrecord(split_X, split_y, tfrecord_path)
                        logger.info(f"折疊 {fold_idx} - 寫入 {count} 個樣本到 {tfrecord_path}")
            
            logger.info(f"成功處理文件: {input_path}")
            return True
        
        except Exception as e:
            logger.error(f"處理文件 {input_path} 時出錯: {e}")
            logger.error(traceback.format_exc())
            return False

    def process_parquet_to_tfrecord(self, input_path: PathLike, output_path: PathLike) -> bool:
        """將Parquet文件直接轉換為TFRecord格式"""
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # 確保目錄存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 加載Parquet
            logger.info(f"加載Parquet文件: {input_path}")
            data = pd.read_parquet(input_path)
            
            # 創建序列
            logger.info(f"創建序列，數據形狀: {data.shape}")
            X_sequences, y_sequences = self.create_sequences(data)
            
            # 寫入TFRecord
            logger.info(f"寫入TFRecord: {output_path}")
            count = self.write_tfrecord(X_sequences, y_sequences, output_path)
            
            logger.info(f"成功寫入 {count} 個樣本到 {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Parquet到TFRecord轉換失敗: {e}")
            logger.error(traceback.format_exc())
            return False

    @log_step("標記充電與放電")
    def mark_charge_discharge(self, data, current_col='current', threshold=0):
        """標記數據為充電和放電階段
        
        Args:
            data: 數據框
            current_col: 電流列名
            threshold: 區分充放電的閾值
            
        Returns:
            pd.DataFrame: 添加充放電標記的數據框
        """
        try:
            # 複製數據以避免修改原始數據
            data = data.copy()
            
            # 嘗試找到匹配的電流列
            if current_col not in data.columns:
                current_cols = [col for col in data.columns 
                            if 'current' in col.lower() and not 'charging' in col.lower()]
                if current_cols:
                    current_col = current_cols[0]
                    logger.info(f"使用 {current_col} 作為電流列")
                else:
                    logger.warning(f"找不到電流列，無法標記充放電階段")
                    return data
            
            # 確保電流列是數值型
            if not pd.api.types.is_numeric_dtype(data[current_col]):
                try:
                    data[current_col] = pd.to_numeric(data[current_col], errors='coerce')
                    logger.info(f"已將 {current_col} 轉換為數值型")
                except Exception as e:
                    logger.warning(f"轉換 {current_col} 為數值型時出錯: {e}")
                    return data
            
            # 標記充電和放電
            data['is_charging'] = data[current_col] <= threshold
            
            # 計算充電和放電數據的比例
            charging_count = data['is_charging'].sum()
            discharging_count = len(data) - charging_count
            charging_pct = charging_count / len(data) * 100 if len(data) > 0 else 0
            
            logger.info(f"數據標記完成: 充電階段 {charging_count} 行 ({charging_pct:.2f}%)，"
                    f"放電階段 {discharging_count} 行 ({100-charging_pct:.2f}%)")
            
            return data
            
        except Exception as e:
            logger.error(f"標記充放電階段時出錯: {e}")
            return data

    @log_step("過濾充電數據")
    def filter_charging_data(self, data, keep_charging=True):
        """過濾充電階段數據，只保留放電數據
        
        Args:
            data: 數據框
            keep_charging: 是否保留充電數據
            
        Returns:
            pd.DataFrame: 過濾後的數據框
        """
        try:
            if 'is_charging' not in data.columns:
                logger.warning("數據未標記充放電階段，先執行標記")
                data = self.mark_charge_discharge(data)
                if 'is_charging' not in data.columns:
                    logger.warning("無法標記充放電階段，跳過過濾")
                    return data
            
            original_count = len(data)
            
            if not keep_charging:
                # 檢查是否需要反轉充放電邏輯
                if self.config.get('invert_charging', False):
                    # 只保留充電數據
                    data = data[data['is_charging']].reset_index(drop=True)
                    filtered_count = original_count - len(data)
                    logger.info(f"已過濾掉 {filtered_count} 行放電數據 ({filtered_count/original_count*100:.2f}%)")
                else:
                    # 只保留放電數據
                    data = data[~data['is_charging']].reset_index(drop=True)
                    filtered_count = original_count - len(data)
                    logger.info(f"已過濾掉 {filtered_count} 行充電數據 ({filtered_count/original_count*100:.2f}%)")
            else:
                logger.info("保留所有數據（充電和放電）")
                
            return data
            
        except Exception as e:
            logger.error(f"過濾充電數據時出錯: {e}")
            return data

class BatchProcessor:
    """電池數據批次處理器，實現BAT文件功能的Python版本"""
    
    def __init__(self):
        """初始化批次處理器"""
        # 獲取目前時間戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 設置基礎路徑為指定的目錄
        self.base_path = Path(r"D:\Document\000-學校\010-中央碩士班\012-鋰電池\20250402\battery_project\cleaned_data")
        
        # 設置輸出目錄和日誌路徑
        self.output_dir = self.base_path / "processed_data"
        self.tfrecord_dir = self.base_path / "tfrecords_combined"
        self.log_path = self.base_path / "logs"
        
        # 確保必要目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tfrecord_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # 創建本次處理的日誌檔案
        self.log_file = self.log_path / f"batch_process_{timestamp}.log"
        
        # 配置日誌處理器，寫入日誌檔案
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"批次處理開始，日誌檔案: {self.log_file}")
        
        # 預處理配置參數
        self.processor_config = {
            'output_dir': str(self.output_dir),
            'tfrecord_dir': str(self.tfrecord_dir),
            'compress_zlib': False,
            'num_parallel': get_optimal_workers(),
            'sequence_length': 60,
            'overlap': 30,
            'test_size': 0.15,
            'val_size': 0.15,
            'fail_safe': True
        }
        
        # 處理選項
        self.temp_folders = []  # 選擇的溫度資料夾
        self.total_processed = 0
        self.total_failed = 0
        self.error_count = 0
    
    def find_parquet_files(self, temp_folder):
        """尋找指定溫度資料夾下的所有parquet檔案"""
        parquet_files = []
        
        # 檢查溫度目錄是否存在
        temp_path = self.base_path / temp_folder
        if not temp_path.exists():
            logger.warning(f"溫度目錄不存在: {temp_path}")
            return parquet_files
        
        # 尋找所有通道資料夾
        channel_folders = []
        for i in range(1, 5):  # 通常通道編號為Ch1到Ch4
            channel_path = temp_path / f"Ch{i}"
            if channel_path.exists():
                channel_folders.append(channel_path)
                logger.info(f"找到通道資料夾: {channel_path}")
        
        # 從每個通道資料夾尋找parquet檔案
        for channel_path in channel_folders:
            files = list(channel_path.glob("*.parquet"))
            parquet_files.extend(files)
            logger.info(f"在 {channel_path} 中找到 {len(files)} 個parquet檔案")
        
        logger.info(f"在 {temp_folder} 溫度下總共找到 {len(parquet_files)} 個parquet檔案")
        return parquet_files
    
    def merge_tfrecords(self, source_files, output_path, dataset_type):
        """合併多個TFRecord檔案"""
        if not source_files:
            logger.warning(f"沒有找到{dataset_type}集的檔案，跳過合併")
            return False
        
        logger.info(f"合併 {len(source_files)} 個 {dataset_type} 檔案到 {output_path}")
        
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果只有一個檔案，直接複製
        if len(source_files) == 1:
            try:
                shutil.copy2(source_files[0], output_path)
                logger.info(f"僅一個檔案，直接複製 {source_files[0]} 到 {output_path}")
                return True
            except Exception as e:
                logger.error(f"複製檔案時出錯: {e}")
                logger.error(traceback.format_exc())
                return False
        
        # 多個檔案情況，使用TF API合併
        compression = "ZLIB" if self.processor_config['compress_zlib'] else ""
        options = tf.io.TFRecordOptions(compression_type=compression)
        
        try:
            with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
                for source_file in source_files:
                    # 讀取每個來源檔案並將記錄寫入合併檔案
                    dataset = tf.data.TFRecordDataset(str(source_file), compression_type=compression)
                    for record in dataset:
                        writer.write(record.numpy())
            
            logger.info(f"成功合併 {len(source_files)} 個檔案到 {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"合併 {dataset_type} 檔案時出錯: {e}")
            logger.error(traceback.format_exc())
            
            # 嘗試備用方法 - 逐個文件內容合併
            try:
                logger.info("嘗試使用備用方法合併檔案")
                with open(output_path, 'wb') as outfile:
                    for source_file in source_files:
                        with open(source_file, 'rb') as infile:
                            outfile.write(infile.read())
                
                logger.info(f"使用備用方法成功合併 {len(source_files)} 個檔案到 {output_path}")
                return True
            except Exception as e2:
                logger.error(f"備用合併方法也失敗: {e2}")
                logger.error(traceback.format_exc())
                
                # 創建一個空的有效文件
                try:
                    with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
                        # 寫入一個空記錄確保文件有效
                        empty_x = np.array([0.0], dtype=np.float32)
                        empty_y = np.array([0.0], dtype=np.float32)
                        feature = {
                            'Xs': tf.train.Feature(float_list=tf.train.FloatList(value=empty_x)),
                            'y': tf.train.Feature(float_list=tf.train.FloatList(value=empty_y))
                        }
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())
                    
                    logger.warning(f"創建了空的TFRecord檔案: {output_path}")
                    return False
                except Exception as e3:
                    logger.error(f"創建空TFRecord檔案也失敗: {e3}")
                    return False
    
    def merge_datasets_for_temp(self, temp_folder):
        """合併特定溫度的數據集，適配 LOOCV"""
        logger.info(f"開始合併 {temp_folder} 溫度的數據集 (LOOCV)")
        
        # 尋找所有訓練、驗證和測試集檔案（包含折疊編號）
        temp_output_dir = self.output_dir / temp_folder
        
        # 使用正則表達式匹配包含 fold 的文件
        train_files = list(temp_output_dir.glob("*_fold*_train.tfrecord"))
        val_files = list(temp_output_dir.glob("*_fold*_val.tfrecord"))
        test_files = list(temp_output_dir.glob("*_fold*_test.tfrecord"))
        
        logger.info(f"找到 {len(train_files)} 個訓練集、{len(val_files)} 個驗證集、{len(test_files)} 個測試集檔案")
        
        # 按折疊分組並合併
        fold_numbers = sorted(set(int(re.search(r'fold(\d+)', f.name).group(1)) for f in train_files + val_files + test_files))
        
        for fold_idx in fold_numbers:
            fold_train_files = [f for f in train_files if f"fold{fold_idx}" in f.name]
            fold_val_files = [f for f in val_files if f"fold{fold_idx}" in f.name]
            fold_test_files = [f for f in test_files if f"fold{fold_idx}" in f.name]
            
            # 合併每個分割
            success_train = self.merge_tfrecords(
                fold_train_files, 
                self.tfrecord_dir / f"{temp_folder}_fold{fold_idx}_combined_train.tfrecord", 
                f"折疊 {fold_idx} 訓練"
            )
            success_val = self.merge_tfrecords(
                fold_val_files, 
                self.tfrecord_dir / f"{temp_folder}_fold{fold_idx}_combined_val.tfrecord", 
                f"折疊 {fold_idx} 驗證"
            )
            success_test = self.merge_tfrecords(
                fold_test_files, 
                self.tfrecord_dir / f"{temp_folder}_fold{fold_idx}_combined_test.tfrecord", 
                f"折疊 {fold_idx} 測試"
            )
        
        # 複製一個縮放器參數文件（假設每個折疊使用相同的標準化參數）
        scaler_files = list(temp_output_dir.glob("*_scaler.parquet"))
        if scaler_files:
            try:
                source_scaler = scaler_files[0]
                target_scaler = self.tfrecord_dir / f"{temp_folder}_combined_scaler.parquet"
                shutil.copy2(source_scaler, target_scaler)
                logger.info(f"複製縮放器參數: {source_scaler} -> {target_scaler}")
                
                json_scaler = source_scaler.with_suffix('.json')
                if json_scaler.exists():
                    target_json = target_scaler.with_suffix('.json')
                    shutil.copy2(json_scaler, target_json)
                    logger.info(f"複製JSON縮放器參數: {json_scaler} -> {target_json}")
            except Exception as e:
                logger.error(f"複製縮放器參數時出錯: {e}")
        
        logger.info(f"完成合併 {temp_folder} 溫度的數據集 (LOOCV)")
        return True
    
    def process_temperature_folder(self, temp_folder):
        """處理一個溫度資料夾下的所有檔案"""
        logger.info(f"開始處理 {temp_folder} 溫度數據")
        print(f"處理 {temp_folder} 溫度數據...")
        
        # 尋找所有parquet檔案
        parquet_files = self.find_parquet_files(temp_folder)
        
        if not parquet_files:
            logger.warning(f"{temp_folder} 溫度未找到檔案，跳過處理")
            print(f"[警告] {temp_folder} 溫度未找到檔案，跳過處理")
            return False
        
        # 確保輸出目錄存在
        temp_output_dir = self.output_dir / temp_folder
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 為每個parquet檔案創建預處理器並處理
        processed_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(parquet_files):
            print(f"[{i+1}/{len(parquet_files)}] 處理中: {file_path.name}")
            
            # 創建輸出前綴
            output_prefix = f"{temp_folder}_{file_path.stem}"
            
            # 創建預處理器
            processor = DataPreprocessor(self.processor_config)
            
            # 處理檔案
            success = processor.process_file(
                file_path, 
                output_prefix=output_prefix, 
                apply_scaling=True
            )
            
            if success:
                processed_count += 1
                print(f"成功處理: {file_path.name}")
            else:
                failed_count += 1
                print(f"[錯誤] 處理 {file_path.name} 失敗")
                
                # 詢問是否繼續處理其他檔案
                if failed_count > 0 and i < len(parquet_files) - 1:
                    continue_choice = input("是否繼續處理其他檔案? (Y/N, 預設Y): ").strip().upper()
                    if continue_choice == "N":
                        logger.info("用戶選擇中止處理")
                        print("用戶選擇中止處理")
                        break
        
        # 合併處理後的數據集
        if processed_count > 0:
            print(f"合併 {temp_folder} 溫度的數據集...")
            self.merge_datasets_for_temp(temp_folder)
        
        logger.info(f"完成處理 {temp_folder} 溫度的 {processed_count} 個檔案，失敗: {failed_count}")
        print(f"完成處理 {temp_folder} 溫度的 {processed_count} 個檔案，失敗: {failed_count}")
        
        self.total_processed += processed_count
        self.total_failed += failed_count
        
        return processed_count > 0
    
    def run_batch_process(self):
        """執行批次處理 (增強進度報告)"""
        logger.info("開始批次處理")
        print("開始批次處理...")
        
        start_time = time.time()
        success_temps = []
        
        try:
            # 計算總工作量
            total_files = 0
            files_by_temp = {}
            for temp_folder in self.temp_folders:
                if self.combined_mode:
                    # 使用整合通道模式
                    success = self.process_temperature_folder_combined(temp_folder)
                else:
                    # 使用單一檔案模式
                    success = self.process_temperature_folder(temp_folder)
                
            if success:
                success_temps.append(temp_folder)
            
            if total_files == 0:
                logger.warning("沒有找到任何檔案需要處理")
                print("沒有找到任何檔案需要處理!")
                return False
            
            print(f"找到 {total_files} 個檔案需要處理，跨 {len(self.temp_folders)} 個溫度")
            
            # 處理進度追蹤
            processed_files = 0
            overall_progress_bar = tqdm(total=total_files, desc="總體進度", position=0)
            
            # 處理每個溫度資料夾
            for temp_idx, temp_folder in enumerate(self.temp_folders):
                files = files_by_temp[temp_folder]
                
                # 使用嵌套進度條顯示當前溫度處理進度
                print(f"\n處理溫度 {temp_folder} ({temp_idx+1}/{len(self.temp_folders)}): {len(files)} 個檔案")
                success = self.process_temperature_folder(temp_folder)
                
                processed_files += len(files)
                overall_progress_bar.update(len(files))
                
                if success:
                    success_temps.append(temp_folder)
                    print(f"✓ 完成溫度 {temp_folder} 處理")
                else:
                    print(f"✗ 溫度 {temp_folder} 處理未完全成功")
            
            overall_progress_bar.close()
            
            # 顯示處理結果
            elapsed_time = time.time() - start_time
            logger.info(f"批次處理完成，耗時 {elapsed_time:.2f} 秒")
            logger.info(f"成功處理 {self.total_processed} 個檔案，失敗 {self.total_failed} 個")
            
            print("\n======================================================")
            if self.total_failed > 0:
                print(f"處理完成但有 {self.total_failed} 個檔案失敗!")
            else:
                print("處理成功完成!")
            print(f"成功處理 {self.total_processed} 個檔案")
            print(f"成功處理的溫度: {success_temps}")
            print(f"處理耗時: {elapsed_time:.2f} 秒")
            print(f"合併的TFRecord檔案已保存到: {self.tfrecord_dir}")
            print(f"詳細日誌已保存到: {self.log_file}")
            print("======================================================")
            
            return self.total_failed == 0
            
        except KeyboardInterrupt:
            logger.info("用戶中斷處理")
            print("\n處理被用戶中斷")
            return False
        except Exception as e:
            logger.error(f"批次處理過程中出錯: {e}")
            logger.error(traceback.format_exc())
            print(f"\n[錯誤] 批次處理過程中出錯: {e}")
            return False

    def setup_interactive(self):
        """互動式設定處理參數 (適配 LOOCV)"""
        print("\n======================================================")
        print("    電池數據處理工具 (Python 批次處理版 - LOOCV 模式)")
        print("======================================================\n")
        
        # 顯示基本路徑資訊
        print(f"基礎路徑: {self.base_path}")
        print(f"輸出目錄: {self.output_dir}")
        print(f"TFRecord目錄: {self.tfrecord_dir}")
        print(f"日誌路徑: {self.log_path}\n")
        
        # 選擇溫度範圍
        print("選擇要處理的溫度範圍:")
        print("[1] 5度數據")
        print("[2] 25度數據")
        print("[3] 45度數據")
        print("[4] 所有溫度數據 (分別處理)")
        
        while True:
            temp_choice = input("請選擇 (1-4): ").strip()
            if temp_choice == "1":
                self.temp_folders = ["5degree"]
                break
            elif temp_choice == "2":
                self.temp_folders = ["25degree"]
                break
            elif temp_choice == "3":
                self.temp_folders = ["45degree"]
                break
            elif temp_choice == "4":
                self.temp_folders = ["5degree", "25degree", "45degree"]
                break
            else:
                print("無效選擇，請選擇1-4之間的數字")
        
        print(f"已選擇溫度: {self.temp_folders}\n")
        logger.info(f"選擇的溫度: {self.temp_folders}")
        
        # 提示 LOOCV 模式
        print("\n數據分割方式:")
        print("將使用留一法交叉驗證 (LOOCV)，每次留出一個電池通道作為測試集，其餘作為訓練集。")
        print("您可以選擇是否從訓練集中劃分驗證集:")
        print("[1] 不劃分驗證集 (所有非測試數據用於訓練)")
        print("[2] 從訓練集中劃分驗證集 (指定比例)")
        
        while True:
            val_choice = input("請選擇 (1-2): ").strip()
            if val_choice == "1":
                self.processor_config['val_size'] = 0.0
                print("已選擇不劃分驗證集")
                logger.info("選擇不劃分驗證集")
                break
            elif val_choice == "2":
                try:
                    val_size = float(input("輸入驗證集比例 (0.0-0.5): ").strip())
                    if not 0.0 <= val_size <= 0.5:
                        print("驗證集比例應在 0.0 到 0.5 之間")
                        continue
                    self.processor_config['val_size'] = val_size
                    print(f"已選擇驗證集比例: {val_size}")
                    logger.info(f"選擇驗證集比例: {val_size}")
                    break
                except ValueError:
                    print("請輸入有效的浮點數")
            else:
                print("無效選擇，請選擇1-2之間的數字")
        
        # 並行處理線程（不變）
        print("\n選擇並行處理線程數:")
        print("[1] 1 線程 (串行處理)")
        print("[2] 2 線程")
        print("[4] 4 線程 (推薦)")
        print("[8] 8 線程 (高效能電腦)")
        
        while True:
            parallel_choice = input("請選擇線程數 (1-8): ").strip()
            if parallel_choice in ["1", "2", "4", "8"]:
                self.processor_config['num_parallel'] = int(parallel_choice)
                print(f"已選擇 {parallel_choice} 線程")
                logger.info(f"選擇 {parallel_choice} 線程")
                break
            else:
                print("無效選擇，使用默認值 4 線程")
                logger.info("無效的線程選擇，使用默認值 4")
                self.processor_config['num_parallel'] = 4
                break
        
        # ZLIB壓縮選項（不變）
        print("\n啟用ZLIB壓縮?")
        print("[Y] 是 (檔案更小，處理更慢)")
        print("[N] 否 (檔案更大，處理更快)")
        
        compress_choice = input("請選擇 (Y/N): ").strip().upper()
        if compress_choice == "Y":
            self.processor_config['compress_zlib'] = True
            print("已啟用ZLIB壓縮")
            logger.info("已啟用ZLIB壓縮")
        else:
            self.processor_config['compress_zlib'] = False
            print("已禁用ZLIB壓縮")
            logger.info("已禁用ZLIB壓縮")
        
        # 序列長度和重疊（不變）
        print("\n設定序列長度和重疊:")
        print("[1] 短序列 (長度=30, 重疊=15)")
        print("[2] 中序列 (長度=60, 重疊=30) [推薦]")
        print("[3] 長序列 (長度=120, 重疊=60)")
        print("[4] 自定義長度")
        
        while True:
            seq_choice = input("請選擇序列設定 (1-4): ").strip()
            if seq_choice == "1":
                self.processor_config['sequence_length'] = 30
                self.processor_config['overlap'] = 15
                print("已選擇短序列 (30/15)")
                logger.info("選擇短序列 (30/15)")
                break
            elif seq_choice == "2":
                self.processor_config['sequence_length'] = 60
                self.processor_config['overlap'] = 30
                print("已選擇中序列 (60/30)")
                logger.info("選擇中序列 (60/30)")
                break
            elif seq_choice == "3":
                self.processor_config['sequence_length'] = 120
                self.processor_config['overlap'] = 60
                print("已選擇長序列 (120/60)")
                logger.info("選擇長序列 (120/60)")
                break
            elif seq_choice == "4":
                try:
                    custom_length = int(input("輸入序列長度: ").strip())
                    custom_overlap = int(input("輸入重疊長度: ").strip())
                    if custom_length <= 0 or custom_overlap < 0 or custom_overlap >= custom_length:
                        print("無效的長度設定，序列長度必須為正數，重疊長度必須小於序列長度")
                        continue
                    self.processor_config['sequence_length'] = custom_length
                    self.processor_config['overlap'] = custom_overlap
                    print(f"已選擇自定義序列設定 ({custom_length}/{custom_overlap})")
                    logger.info(f"選擇自定義序列設定 ({custom_length}/{custom_overlap})")
                    break
                except ValueError:
                    print("請輸入有效的整數")
            else:
                print("無效選擇，請選擇1-4之間的數字")
        
        # 添加處理模式選擇
        print("\n選擇處理模式:")
        print("[1] 單一檔案模式 (逐一處理每個檔案)")
        print("[2] 整合通道模式 (LOOCV，每個通道作為一個測試集) [推薦]")
        
        while True:
            mode_choice = input("請選擇處理模式 (1-2): ").strip()
            if mode_choice == "1":
                self.combined_mode = False
                print("已選擇單一檔案模式")
                logger.info("選擇單一檔案模式")
                break
            elif mode_choice == "2":
                self.combined_mode = True
                print("已選擇整合通道模式 (LOOCV)")
                logger.info("選擇整合通道模式 (LOOCV)")
                break
            else:
                print("無效選擇，請選擇1-2之間的數字")
        
        # 添加充放電處理選項
        print("\n處理充放電數據:")
        print("[1] 保留所有數據（充電和放電）")
        print("[2] 只保留放電數據")
        print("[3] 只保留充電數據")

        while True:
            charge_choice = input("請選擇充放電處理方式 (1-3): ").strip()
            if charge_choice == "1":
                self.processor_config['process_charge_discharge'] = True
                self.processor_config['filter_charging'] = False
                print("已選擇保留所有數據")
                logger.info("選擇保留所有數據（充電和放電）")
                break
            elif charge_choice == "2":
                self.processor_config['process_charge_discharge'] = True
                self.processor_config['filter_charging'] = True
                self.processor_config['charge_threshold'] = 0
                print("已選擇只保留放電數據")
                logger.info("選擇只保留放電數據")
                break
            elif charge_choice == "3":
                self.processor_config['process_charge_discharge'] = True
                self.processor_config['filter_charging'] = True
                self.processor_config['invert_charging'] = True  # 新增參數，用於反轉充放電邏輯
                self.processor_config['charge_threshold'] = 0
                print("已選擇只保留充電數據")
                logger.info("選擇只保留充電數據")
                break
            else:
                print("無效選擇，請選擇1-3之間的數字")

        # 如果選擇處理充放電數據，詢問閾值
        if self.processor_config.get('process_charge_discharge', False):
            # 詢問閾值
            try:
                threshold = float(input("輸入充放電閾值 (預設為 0): ").strip() or "0")
                self.processor_config['charge_threshold'] = threshold
                print(f"已設定充放電閾值為 {threshold}")
                logger.info(f"設定充放電閾值為 {threshold}")
            except ValueError:
                print("無效的閾值，使用預設值 0")
                self.processor_config['charge_threshold'] = 0
                logger.info("使用預設充放電閾值 0")

        # 顯示處理參數摘要
        print("\n處理參數摘要:")
        print(f"- 處理路徑: {self.base_path}")
        print(f"- 輸出路徑: {self.output_dir}")
        print(f"- TFRecord輸出路徑: {self.tfrecord_dir}")
        print(f"- 處理溫度: {self.temp_folders}")
        print(f"- 數據分割: 使用 LOOCV，驗證集比例={self.processor_config['val_size']:.2f}")
        print(f"- 序列設置: 長度={self.processor_config['sequence_length']}, 重疊={self.processor_config['overlap']}")
        print(f"- 並行線程: {self.processor_config['num_parallel']}")
        print(f"- ZLIB壓縮已啟用: {'是' if self.processor_config['compress_zlib'] else '否'}")
        print(f"- 失敗安全模式已啟用: 是")
        
        logger.info("參數摘要:")
        logger.info(f"- 處理溫度: {self.temp_folders}")
        logger.info(f"- 數據分割: 使用 LOOCV，驗證集比例={self.processor_config['val_size']:.2f}")
        logger.info(f"- 序列設置: 長度={self.processor_config['sequence_length']}, 重疊={self.processor_config['overlap']}")
        logger.info(f"- 並行線程: {self.processor_config['num_parallel']}")
        logger.info(f"- ZLIB壓縮已啟用: {'是' if self.processor_config['compress_zlib'] else '否'}")
        logger.info(f"- 失敗安全模式已啟用: 是")
        
        # 確認開始處理
        print("\n準備開始處理?")
        confirm = input("確認開始 (Y/N): ").strip().upper()
        if confirm != "Y":
            print("操作已取消")
            logger.info("用戶取消操作")
            sys.exit(0)
        
        print("\n開始電池數據處理和通道合併 (LOOCV 模式)...")
        print(f"處理日誌將保存到: {self.log_file}")
        print()

    def process_temperature_folder_combined(self, temp_folder):
        """處理一個溫度資料夾下的所有通道，組合成LOOCV數據集"""
        logger.info(f"開始以LOOCV模式處理 {temp_folder} 溫度下的所有通道數據")
        print(f"處理 {temp_folder} 溫度數據 (LOOCV整合模式)...")
        
        # 獲取該溫度下所有通道資料夾
        temp_path = self.base_path / temp_folder
        channel_folders = []
        for i in range(1, 5):  # 通常通道編號為Ch1到Ch4
            channel_path = temp_path / f"Ch{i}"
            if channel_path.exists():
                channel_folders.append(channel_path)
                logger.info(f"找到通道資料夾: {channel_path}")
        
        if not channel_folders:
            logger.warning(f"{temp_folder} 溫度下未找到通道資料夾，跳過處理")
            print(f"[警告] {temp_folder} 溫度下未找到通道資料夾，跳過處理")
            return False
        
        # 為每個通道加載所有數據
        channel_data = {}
        channel_files_count = {}
        
        for channel_path in channel_folders:
            channel_name = channel_path.name  # 例如 "Ch1"
            parquet_files = list(channel_path.glob("*.parquet"))
            
            if not parquet_files:
                logger.warning(f"通道 {channel_name} 未找到parquet檔案，跳過")
                continue
            
            logger.info(f"通道 {channel_name} 找到 {len(parquet_files)} 個檔案")
            channel_files_count[channel_name] = len(parquet_files)
            
            # 加載所有檔案並合併
            channel_dfs = []
            for file_path in tqdm(parquet_files, desc=f"載入 {channel_name} 檔案"):
                try:
                    # 創建預處理器
                    processor = DataPreprocessor(self.processor_config)
                    # 載入數據
                    df = processor.load_and_clean_data(file_path)
                    # 添加通道標識
                    df['channel'] = channel_name
                    channel_dfs.append(df)
                except Exception as e:
                    logger.error(f"處理檔案 {file_path.name} 失敗: {e}")
                    continue
            
            if channel_dfs:
                # 合併該通道的所有數據框
                channel_data[channel_name] = pd.concat(channel_dfs, ignore_index=True)
                logger.info(f"通道 {channel_name} 合併了 {len(channel_dfs)} 個檔案，共 {len(channel_data[channel_name])} 行")
        
        if not channel_data:
            logger.warning(f"{temp_folder} 溫度下無法加載任何數據，跳過處理")
            print(f"[警告] {temp_folder} 溫度下無法加載任何數據，跳過處理")
            return False
        
        # 按溫度區間分割處理
        # 為每個溫度區間執行LOOCV
        for channel_name, df in channel_data.items():
            # 使用預處理器按溫度分割
            processor = DataPreprocessor(self.processor_config)
            temp_dfs = processor.split_by_temperature(df)
            
            for temp_label, temp_df in temp_dfs.items():
                # 為每個溫度標籤創建輸出前綴
                output_prefix = f"{temp_folder}_{temp_label}_combined"
                
                # 使用LOOCV方法處理和生成數據集
                self.process_loocv_datasets(channel_data, temp_label, output_prefix)
        
        return True

def process_loocv_datasets(self, channel_data, temp_label, output_prefix):
    """使用LOOCV方法處理數據並生成訓練/測試集"""
    # 使用channel_data中的數據為每個通道創建LOOCV分割
    
    # 獲取所有通道名稱
    channel_names = list(channel_data.keys())
    
    # 循環每個通道作為測試集
    for test_channel in channel_names:
        # 創建一個fold_idx，基於通道編號
        fold_idx = int(test_channel.replace("Ch", "")) - 1  # Ch1->0, Ch2->1, etc.
        
        # 分割數據：測試集為當前通道，訓練集為其他通道
        test_data = channel_data[test_channel]
        train_channels = [ch for ch in channel_names if ch != test_channel]
        
        if not train_channels:
            logger.warning(f"沒有足夠的通道用於訓練，跳過 {test_channel}")
            continue
        
        # 合併訓練數據
        train_dfs = [channel_data[ch] for ch in train_channels]
        train_data = pd.concat(train_dfs, ignore_index=True)
        
        # 創建序列
        processor = DataPreprocessor(self.processor_config)
        
        # 處理訓練數據
        logger.info(f"處理 {test_channel} 作為測試集的LOOCV折疊")
        X_train, y_train = processor.create_sequences(train_data)
        
        # 處理測試數據
        X_test, y_test = processor.create_sequences(test_data)
        
        # 計算標準化參數（僅使用訓練數據）
        scaler_params = processor.compute_scaler_params(X_train, y_train)
        
        # 保存標準化參數
        scaler_path = Path(self.output_dir) / f"{output_prefix}_fold{fold_idx}_scaler.parquet"
        processor.save_scaler_params(scaler_params, scaler_path)
        
        # 應用標準化
        X_train_scaled, y_train_scaled = processor.apply_scaling(X_train, y_train, scaler_params)
        X_test_scaled, y_test_scaled = processor.apply_scaling(X_test, y_test, scaler_params)
        
        # 從訓練數據中分割驗證集（如果需要）
        val_size = self.processor_config.get('val_size', 0.15)
        if val_size > 0:
            # 隨機分割驗證集
            indices = np.arange(X_train_scaled.shape[0])
            np.random.shuffle(indices)
            val_count = int(len(indices) * val_size)
            
            val_indices = indices[:val_count]
            train_indices = indices[val_count:]
            
            X_val, y_val = X_train_scaled[val_indices], y_train_scaled[val_indices]
            X_train_final, y_train_final = X_train_scaled[train_indices], y_train_scaled[train_indices]
        else:
            # 不使用驗證集
            X_train_final, y_train_final = X_train_scaled, y_train_scaled
            X_val, y_val = np.array([]), np.array([])
        
        # 寫入TFRecord
        tfrecord_dir = Path(self.tfrecord_dir)
        
        # 訓練集
        train_path = tfrecord_dir / f"{output_prefix}_fold{fold_idx}_train.tfrecord"
        train_count = processor.write_tfrecord(X_train_final, y_train_final, train_path)
        logger.info(f"折疊 {fold_idx} - 寫入 {train_count} 個樣本到 {train_path}")
        
        # 驗證集（如果有）
        if X_val.size > 0:
            val_path = tfrecord_dir / f"{output_prefix}_fold{fold_idx}_val.tfrecord"
            val_count = processor.write_tfrecord(X_val, y_val, val_path)
            logger.info(f"折疊 {fold_idx} - 寫入 {val_count} 個樣本到 {val_path}")
        
        # 測試集
        test_path = tfrecord_dir / f"{output_prefix}_fold{fold_idx}_test.tfrecord"
        test_count = processor.write_tfrecord(X_test_scaled, y_test_scaled, test_path)
        logger.info(f"折疊 {fold_idx} - 寫入 {test_count} 個樣本到 {test_path}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="電池數據預處理腳本")
    
    # 輸入輸出參數
    parser.add_argument("--input", "-i", type=str, required=False, help="輸入檔案路徑")
    parser.add_argument("--output-prefix", "-o", type=str, default=None, help="輸出檔案前綴")
    parser.add_argument("--data-dir", type=str, default=config.get("system.data_dir", "data"), help="數據目錄")
    parser.add_argument("--output-dir", type=str, default=config.get("system.output_dir", "output"), help="輸出目錄")
    parser.add_argument("--tfrecord-dir", type=str, default=config.get("system.tfrecord_dir", "tfrecords"), help="TFRecord目錄")
    parser.add_argument("--cache-dir", type=str, default=config.get("system.cache_dir", "cache"), help="快取目錄")
    
    # 列映射參數
    parser.add_argument("--column-map", type=str, default=None,
                      help="列映射，格式為'標準列名:原始列名'，多個映射用逗號分隔。特殊值'auto'表示自動派生")
    
    # 處理選項
    parser.add_argument("--seq-length", "-sl", type=int, default=config.get("training.sequence_length"), help="序列長度")
    parser.add_argument("--overlap", type=int, default=None, help="序列重疊長度")
    parser.add_argument("--no-scaling", action="store_true", help="不應用標準化")
    parser.add_argument("--compress", "-z", action="store_true", help="使用ZLIB壓縮TFRecord")
    parser.add_argument("--chunk-size", type=int, default=50000, help="處理塊大小")
    parser.add_argument("--to-tfrecord", "-t", action="store_true", help="僅轉換Parquet到TFRecord")
    parser.add_argument("--parallel", "-p", type=int, default=None, help="並行處理線程數")
    parser.add_argument("--use-float16", action="store_true", help="使用半精度浮點數以節省記憶體")
    parser.add_argument("--safe-mode", action="store_true", help="安全模式（更嚴格的錯誤檢查）")
    
    # 資料分割選項
    parser.add_argument("--test-size", type=float, default=config.get("data.test_size"), help="測試集比例")
    parser.add_argument("--val-size", type=float, default=config.get("data.val_size"), help="驗證集比例")
    parser.add_argument("--random-seed", type=int, default=config.get("data.random_seed"), help="隨機種子")
    
    # 顯示模式選項
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細輸出模式")
    parser.add_argument("--quiet", "-q", action="store_true", help="靜默模式")
    
    # 添加安全模式參數
    parser.add_argument("--fail-safe", action="store_true", help="失敗安全模式：出錯時不中止而是嘗試繼續")
    
    try:
        args = parser.parse_args()
        
        # 設置日誌級別
        if args.quiet:
            logger.setLevel(logging.WARNING)
        elif args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # 如果沒有提供輸入檔案，顯示幫助
        if not args.input and not args.to_tfrecord:
            parser.print_help()
            return 1
        
        # 配置覆蓋
        config_override = {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'tfrecord_dir': args.tfrecord_dir,
            'cache_dir': args.cache_dir,  # 添加 cache_dir
            'sequence_length': args.seq_length,
            'compress_zlib': args.compress,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'random_seed': args.random_seed,
            'chunk_size': args.chunk_size,
            'use_float16': args.use_float16,
            'safe_mode': args.safe_mode,
            'fail_safe': args.fail_safe if hasattr(args, 'fail_safe') else False
        }
        
        # 處理列映射
        if args.column_map:
            column_map = {}
            mappings = args.column_map.split(',')
            for mapping in mappings:
                parts = mapping.split(':')
                if len(parts) == 2:
                    std_col, orig_col = parts
                    column_map[std_col] = orig_col
            
            if column_map:
                config_override['column_map'] = column_map
                logger.info(f"應用自定義列映射: {column_map}")
        
        # 如果指定了重疊長度
        if args.overlap is not None:
            config_override['overlap'] = args.overlap
        
        # 如果指定了並行線程數
        if args.parallel is not None:
            config_override['num_parallel'] = args.parallel
        
        # 創建預處理器
        try:
            preprocessor = DataPreprocessor(config_override)
        except Exception as e:
            logger.error(f"創建預處理器時出錯: {e}")
            logger.error(traceback.format_exc())
            # 嘗試使用最小配置重新創建
            try:
                logger.warning("嘗試使用最小配置重新創建預處理器...")
                minimal_config = {
                    'data_dir': args.data_dir,
                    'output_dir': args.output_dir,
                    'tfrecord_dir': args.tfrecord_dir,
                    'num_parallel': 1,  # 減少並行度
                    'safe_mode': True   # 啟用安全模式
                }
                preprocessor = DataPreprocessor(minimal_config)
            except Exception as e2:
                logger.error(f"使用最小配置創建預處理器也失敗: {e2}")
                return 1
        
        try:
            # 記錄起始記憶體使用
            memory_manager.log_memory_usage("預處理開始")
            
            if args.to_tfrecord:
                # 僅執行Parquet到TFRecord的轉換
                if not args.input:
                    logger.error("缺少輸入檔案路徑")
                    return 1
                
                if not args.input.endswith('.parquet'):
                    logger.error("輸入檔案必須是Parquet格式")
                    return 1
                
                # 確定輸出路徑
                if args.output_prefix:
                    output_path = Path(args.tfrecord_dir) / f"{args.output_prefix}.tfrecord"
                else:
                    base_name = Path(args.input).stem
                    output_path = Path(args.tfrecord_dir) / f"{base_name}.tfrecord"
                
                # 執行轉換
                try:
                    success = preprocessor.process_parquet_to_tfrecord(args.input, output_path)
                    
                    if success:
                        logger.info(f"轉換成功: {args.input} → {output_path}")
                        return 0
                    else:
                        logger.error(f"轉換失敗")
                        return 1
                except Exception as e:
                    logger.error(f"執行Parquet到TFRecord轉換時出錯: {e}")
                    logger.error(traceback.format_exc())
                    
                    # 在失敗安全模式下，嘗試創建一個空的輸出文件
                    if config_override.get('fail_safe', False):
                        try:
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            options = tf.io.TFRecordOptions(
                                compression_type="ZLIB" if args.compress else ""
                            )
                            with tf.io.TFRecordWriter(str(output_path), options=options) as writer:
                                pass
                            logger.warning(f"轉換失敗，但已創建空TFRecord文件: {output_path}")
                            return 1  # 仍然返回錯誤碼
                        except Exception as e2:
                            logger.error(f"創建空TFRecord文件失敗: {e2}")
                    
                    return 1
            else:
                # 執行完整的數據預處理
                try:
                    preprocessor.process_file(
                        args.input,
                        output_prefix=args.output_prefix,
                        apply_scaling=not args.no_scaling
                    )
                    
                    # 記錄結束記憶體使用
                    memory_manager.log_memory_usage("預處理完成")
                    return 0
                except Exception as e:
                    logger.error(f"處理文件時出錯: {e}")
                    logger.error(traceback.format_exc())
                    
                    # 在失敗安全模式下，嘗試保存部分結果
                    if config_override.get('fail_safe', False):
                        logger.warning("在失敗安全模式下嘗試保存部分結果...")
                        try:
                            # 確保輸出目錄存在
                            output_dir = Path(config_override['output_dir'])
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # 創建一個指示失敗的標記文件
                            with open(output_dir / f"{args.output_prefix if args.output_prefix else 'unknown'}_FAILED.txt", "w") as f:
                                f.write(f"處理失敗時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"錯誤信息: {str(e)}\n")
                            
                            logger.info("已創建失敗標記文件")
                        except Exception as e2:
                            logger.error(f"創建失敗標記文件時出錯: {e2}")
                    
                    return 1
                
        except Exception as e:
            logger.error(f"預處理失敗: {e}")
            logger.error(traceback.format_exc())
            return 1
            
    except Exception as e:
        # 捕獲參數解析或其他早期錯誤
        print(f"錯誤: {e}")
        traceback.print_exc()
        return 1
    
    finally:
        # 確保在任何情況下都進行記憶體清理
        try:
            memory_manager.collect_garbage()
            logger.info("已執行最終記憶體清理")
        except:
            pass


# 將這段程式碼替換到原始 preprocess_data.py 檔案的最末尾部分

# 程式入口點
if __name__ == "__main__":
    try:
        # 明確輸出確認程式已啟動
        print("\n程式啟動中，請稍候...\n")
        
        # 檢查是否需要直接執行批次處理（不帶任何參數）
        if len(sys.argv) == 1:
            print("未提供參數，啟動互動式批次處理模式...")
            
            # 增加的調試資訊
            print("正在初始化批次處理器...")
            
            try:
                batch_processor = BatchProcessor()
                print("批次處理器初始化完成。")
                print("正在顯示互動式選單...")
                batch_processor.setup_interactive()
                print("互動式設定完成，正在執行批次處理...")
                success = batch_processor.run_batch_process()
                print("批次處理已完成。")
                # 等待用戶按鍵，防止視窗立即關閉
                input("按任意鍵繼續...")
                sys.exit(0 if success else 1)
            except Exception as e:
                print(f"初始化批次處理器出錯: {e}")
                traceback.print_exc()
                # 等待用戶按鍵，防止視窗立即關閉
                input("按任意鍵繼續...")
                sys.exit(1)
        else:
            # 執行命令列參數解析和處理
            exit_code = main()
            sys.exit(exit_code)
    except Exception as e:
        print(f"嚴重錯誤: {e}")
        traceback.print_exc()
        # 等待用戶按鍵，防止視窗立即關閉
        input("按任意鍵繼續...")
        sys.exit(1)