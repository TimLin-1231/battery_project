#scripts/clean_data.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
電池數據清理與處理系統 (優化版)

提供全面的電池數據清理、轉換與錯誤檢測功能，使用管道模式實現高效的數據處理流程
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar, Tuple, Set
from functools import wraps
import traceback
import re
import time
import logging
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor


# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# 導入項目模塊
try:
    from config.base_config import config
    from core.logging import setup_logger, LoggingTimer
    # 導入共用模組
    from utils import setup_logger, extract_temp_and_channel
    HAS_CUSTOM_LOGGING = True
except ImportError:
    # 如果沒有自定義日誌系統，使用標準日誌
    HAS_CUSTOM_LOGGING = False
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

# 設置日誌
if HAS_CUSTOM_LOGGING:
    logger = setup_logger("clean_data")
else:
    logger = logging.getLogger("clean_data")

# 禁用 pandas 警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 類型變量定義
T = TypeVar('T')
PathLike = Union[str, Path]
DataFrame = pd.DataFrame

# ============================================================================
# 欄位映射與單位常數
# ============================================================================

# 必要欄位與欄位別名映射
REQUIRED_COLUMNS = [
    'time', 'voltage', 'current', 'temp'
]

COLUMN_ALIASES = {
    # 時間列映射 - 您的數據使用 DateTime
    'datetime': 'time',
    'DateTime': 'time',  # 加入實際列名，確保大小寫正確匹配
    'date_time': 'time',
    'timestamp': 'time',
    
    # 電壓列映射 - 您的數據使用 BMS_PackVoltage
    'BMS_PackVoltage': 'voltage',
    'packvoltage': 'voltage',
    'bms_packvoltage': 'voltage',
    'bms_packvoltage_v': 'voltage',
    'bms_packvoltage_mv': 'voltage',
    
    # 電流列映射 - 您的數據使用 BMS_PackCurrent
    'BMS_PackCurrent': 'current',
    'packcurrent': 'current',
    'bms_packcurrent': 'current',
    'bms_packcurrent_a': 'current',
    
    # 溫度列映射 - 您的數據使用 BMS_Temp1 和 BMS_Temp2
    'BMS_Temp1': 'temp',
    'BMS_Temp2': 'bms_temp2',
    'temperature': 'temp',
    'bms_temp1': 'temp',
    'bms_temperature': 'temp',
    
    # 其他BMS相關欄位映射
    'BMS_MOSTemp1': 'bms_mostemp1',
    'BMS_MOSTemp2': 'bms_mostemp2',
    'mostemp1': 'bms_mostemp1',
    'mostemp2': 'bms_mostemp2',
    
    'BMS_CycleCount': 'bms_cyclecount',
    'cyclecount': 'bms_cyclecount',
    'cycle': 'bms_cyclecount',
    'cycle_count': 'bms_cyclecount',
    
    'BMS_StateOfHealth': 'bms_stateofhealth',
    'stateofhealth': 'bms_stateofhealth',
    'soh': 'bms_stateofhealth',
    
    'BMS_FDCR': 'bms_fdcr',
    'fdcr': 'bms_fdcr',
    
    'BMS_RSOC': 'bms_rsoc',
    'rsoc': 'bms_rsoc',
    'soc': 'bms_rsoc',
    
    'BMS_ASOC': 'bms_asoc',
    'asoc': 'bms_asoc',
    
    'BMS_AvgCurrent': 'bms_avgcurrent',
    'avgcurrent': 'bms_avgcurrent',
    'bms_avgcurrent_a': 'bms_avgcurrent',
    
    'BMS_ChargingCurrent': 'bms_chargingcurrent',
    'chargingcurrent': 'bms_chargingcurrent',
    'bms_chargingcurrent_a': 'bms_chargingcurrent',
    
    'BMS_DCR': 'bms_dcr',
    'dcr': 'bms_dcr',
    
    'BMS_RC': 'bms_rc',
    'rc': 'bms_rc',
    'bms_rc_ah': 'bms_rc',
    
    'BMS_FCC': 'bms_fcc',
    'fcc': 'bms_fcc',
    'bms_fcc_ah': 'bms_fcc',
    
    'BMS_WarnStatus': 'bms_warnstatus',
    'BMS_BatteryStatus': 'bms_batterystatus',
    
    # 單元電壓映射
    'BMS_CellVolt1': 'bms_cellvolt1',
    'BMS_CellVolt2': 'bms_cellvolt2',
    'BMS_CellVolt3': 'bms_cellvolt3',
    'BMS_CellVolt4': 'bms_cellvolt4',
    'BMS_CellVolt5': 'bms_cellvolt5',
    'BMS_CellVolt6': 'bms_cellvolt6',
    'BMS_CellVolt7': 'bms_cellvolt7',
    'BMS_CellVolt8': 'bms_cellvolt8',
    'BMS_CellVolt9': 'bms_cellvolt9',
    'BMS_CellVolt10': 'bms_cellvolt10',
    'BMS_CellVolt11': 'bms_cellvolt11',
    'BMS_CellVolt12': 'bms_cellvolt12',
    'BMS_CellVolt13': 'bms_cellvolt13',
    'bms_cellvolt1': 'bms_cellvolt1',
    'bms_cellvolt2': 'bms_cellvolt2',
    'bms_cellvolt3': 'bms_cellvolt3',
    'bms_cellvolt4': 'bms_cellvolt4',
    'bms_cellvolt5': 'bms_cellvolt5',
    'bms_cellvolt6': 'bms_cellvolt6',
    'bms_cellvolt7': 'bms_cellvolt7',
    'bms_cellvolt8': 'bms_cellvolt8',
    'bms_cellvolt9': 'bms_cellvolt9',
    'bms_cellvolt10': 'bms_cellvolt10',
    'bms_cellvolt11': 'bms_cellvolt11',
    'bms_cellvolt12': 'bms_cellvolt12',
    'bms_cellvolt13': 'bms_cellvolt13',
}

# 單位轉換閾值
UNIT_THRESHOLDS = {
    'voltage': {'threshold': 100, 'from': 'mV', 'to': 'V', 'factor': 0.001},
    'current': {'threshold': 100, 'from': 'mA', 'to': 'A', 'factor': 0.001},
    'dcr': {'threshold': 1000, 'from': 'μΩ', 'to': 'Ω', 'factor': 1e-6},
    'fdcr': {'threshold': 1000, 'from': 'μΩ', 'to': 'Ω', 'factor': 1e-6},
    'rc': {'threshold': 100, 'from': 'mAh', 'to': 'Ah', 'factor': 0.001},
    'fcc': {'threshold': 100, 'from': 'mAh', 'to': 'Ah', 'factor': 0.001},
}

# 物理量範圍約束
PHYSICAL_CONSTRAINTS = {
    'BMS_PackVoltage': {'min': 2500, 'max': 54000, 'unit': 'mV'},  
    'BMS_PackCurrent': {'min': -7000, 'max': 7000, 'unit': 'mA'},
    'BMS_AvgCurrent': {'min': -7000, 'max': 7000, 'unit': 'mA'},
    'BMS_Temp1': {'min': -10, 'max': 60, 'unit': '°C'},
    'BMS_Temp2': {'min': -10, 'max': 60, 'unit': '°C'},
    'BMS_MOSTemp1': {'min': -10, 'max': 60, 'unit': '°C'},
    'BMS_MOSTemp2': {'min': -10, 'max': 60, 'unit': '°C'},
    'BMS_RSOC': {'min': 0, 'max': 100, 'unit': '%'},
    'BMS_ASOC': {'min': 0, 'max': 100, 'unit': '%'},
    'BMS_StateOfHealth': {'min': 0, 'max': 100, 'unit': '%'},
    'BMS_CycleCount': {'min': 0, 'max': 1000, 'unit': 'cycles'},
    'BMS_CellVolt1': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt2': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt3': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt4': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt5': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt6': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt7': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt8': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt9': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt10': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt11': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt12': {'min': 2500, 'max': 4200, 'unit': 'mV'},
    'BMS_CellVolt13': {'min': 2500, 'max': 4200, 'unit': 'mV'},
}

# 儲存清理報告與失敗檔案
CLEANING_REPORT = []
FAILED_FILES = []

def log_step(step_name: str):
    """記錄數據處理步驟的裝飾器
    
    Args:
        step_name: 步驟名稱
    """
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

# ============================================================================
# 工具函數
# ============================================================================

def robust_datetime_parsing(df, datetime_col='time'):
    """
    解析多種日期時間格式，失敗則使用預設值。
    
    Args:
        df: 數據框
        datetime_col: 時間列名
    
    Returns:
        添加解析時間戳的數據框
    """
    df = df.copy()
    
    if datetime_col not in df.columns:
        logger.warning(f"找不到列 {datetime_col}，使用默認時間。")
        df['parsed_dt'] = pd.to_datetime('2025-01-01 00:00:00')
        return df

    # 確保時間列是字符串型或已經是日期時間型
    if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        # 已經是日期時間型，直接複製
        df['parsed_dt'] = df[datetime_col]
        return df
    elif not pd.api.types.is_string_dtype(df[datetime_col]):
        try:
            df[datetime_col] = df[datetime_col].astype(str)
        except:
            logger.warning(f"無法將 {datetime_col} 轉換為字符串類型，使用默認時間。")
            df['parsed_dt'] = pd.to_datetime('2025-01-01 00:00:00')
            return df

    formats = [
        '%Y/%m/%d %p %I:%M:%S',  
        '%Y/%m/%d %I:%M:%S %p',
        '%Y-%m-%d %I:%M:%S %p',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S.%f',
        '%d/%m/%Y %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%Y%m%d %H:%M:%S'
    ]
    
    parsed = None
    successfully_parsed = 0
    
    for fmt in formats:
        try:
            parsed = pd.to_datetime(df[datetime_col], format=fmt, errors='coerce')
            successfully_parsed = (~parsed.isna()).sum()
            if successfully_parsed > 0:
                logger.info(f"成功使用格式 {fmt} 解析 {datetime_col} 中的 {successfully_parsed} 行。")
                # 如果大部分都解析成功，就不再嘗試其他格式
                if successfully_parsed / len(parsed) > 0.9:
                    break
        except (ValueError, TypeError):
            continue

    if parsed is None or parsed.isna().all():
        logger.warning(f"所有預定義格式均無法解析 {datetime_col}，嘗試推斷格式。")
        try:
            parsed = pd.to_datetime(df[datetime_col], errors='coerce')
            successfully_parsed = (~parsed.isna()).sum()
            if successfully_parsed > 0:
                logger.info(f"成功推斷 {datetime_col} 的日期時間格式，解析了 {successfully_parsed} 行。")
            else:
                sample_data = df[datetime_col].dropna().head().tolist()
                logger.error(f"無法推斷日期時間格式。樣本數據: {sample_data}")
                parsed = pd.Series([pd.Timestamp('2025-01-01 00:00:00')] * len(df), index=df.index)
        except Exception as e:
            logger.error(f"日期時間推斷過程中出錯: {e}。使用默認值。")
            parsed = pd.Series([pd.Timestamp('2025-01-01 00:00:00')] * len(df), index=df.index)

    df['parsed_dt'] = parsed
    na_count = df['parsed_dt'].isna().sum()
    if na_count > 0:
        logger.info(f"解析 {datetime_col} 後，發現 {na_count} 個 NaT 值，將填充缺失值。")
        # 使用前向和後向填充
        df['parsed_dt'] = df['parsed_dt'].fillna(method='ffill').fillna(method='bfill')
        # 如果仍有缺失值，使用默認值
        remaining_na = df['parsed_dt'].isna().sum()
        if remaining_na > 0:
            df.loc[df['parsed_dt'].isna(), 'parsed_dt'] = pd.Timestamp('2025-01-01 00:00:00')
            logger.warning(f"仍有 {remaining_na} 個無法解析的時間值，使用默認值替代。")

    return df

def fix_timestamps(df, dt_col='parsed_dt', out_col='timeindex', min_dt_threshold=1e-3):
    """
    修復時間戳並生成連續時間索引。
    
    Args:
        df: 數據框
        dt_col: 日期時間列名
        out_col: 輸出列名
        min_dt_threshold: 最小時間間隔閾值
        
    Returns:
        修復時間戳後的數據框
    """
    df = df.copy()
    if dt_col not in df.columns or df.empty:
        logger.warning(f"找不到列 {dt_col} 或數據框為空，返回默認值。")
        df[out_col] = 0.0
        return df

    # 確保日期時間列是正確的類型
    try:
        df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
    except Exception as e:
        logger.error(f"轉換 {dt_col} 列為日期時間類型時出錯: {e}")
        df[out_col] = 0.0
        return df
        
    # 處理缺失值
    na_count = df[dt_col].isna().sum()
    if na_count > 0:
        logger.warning(f"{dt_col} 列中有 {na_count} 個缺失值，將被過濾")
        df = df.dropna(subset=[dt_col])
        if df.empty:
            logger.warning("所有時間戳無效，返回默認值。")
            df[out_col] = 0.0
            return df

    # 確保按時間排序
    try:
        df = df.sort_values(dt_col)
    except Exception as e:
        logger.error(f"按時間戳排序時出錯: {e}")
        
    # 計算相對時間索引（秒）
    try:
        df[out_col] = (df[dt_col] - df[dt_col].iloc[0]).dt.total_seconds()
    except Exception as e:
        logger.error(f"計算時間索引時出錯: {e}")
        df[out_col] = 0.0
        return df
        
    # 檢查是否有重複的時間戳
    if df[out_col].nunique() == 1:
        logger.warning(f"所有時間戳相同 ({df[dt_col].iloc[0]})，跳過時間差過濾。")
        return df

    # 過濾過近的時間戳
    try:
        mask = df[out_col].diff().fillna(min_dt_threshold) >= min_dt_threshold
        filtered_df = df[mask].copy()
        if filtered_df.empty:
            logger.warning("過濾後沒有有效時間戳，返回原始數據。")
            return df
        else:
            filtered_count = len(df) - len(filtered_df)
            if filtered_count > 0:
                logger.info(f"已修復時間戳，過濾掉 {filtered_count} 行數據 ({filtered_count/len(df)*100:.1f}%)。")
            return filtered_df
    except Exception as e:
        logger.error(f"過濾時間戳時出錯: {e}")
        return df

def read_file_with_recovery(file_path, chunksize=50000):
    """
    分塊讀取檔案，支持 CSV 和 Excel 格式，具有錯誤恢復能力。
    
    Args:
        file_path: 文件路徑
        chunksize: 分塊大小
        
    Returns:
        數據框或 None（讀取失敗時）
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return None
        
    try:
        ext = file_path.suffix.lower()
        if ext == '.csv':
            # 嘗試不同的讀取策略和編碼
            encodings = ['utf-8', 'latin1', 'cp1252', 'gb18030']
            
            # 先嘗試讀取前幾行以檢測編碼和分隔符
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        sample = ''.join([f.readline() for _ in range(5)])
                    
                    # 根據樣本檢測分隔符
                    if sample.count(',') > sample.count(';') and sample.count(',') > sample.count('\t'):
                        sep = ','
                    elif sample.count(';') > sample.count(',') and sample.count(';') > sample.count('\t'):
                        sep = ';'
                    elif sample.count('\t') > sample.count(',') and sample.count('\t') > sample.count(';'):
                        sep = '\t'
                    else:
                        sep = ','  # 默認分隔符
                        
                    logger.info(f"檢測到分隔符: '{sep}'，使用編碼: {encoding}")
                    break
                except Exception:
                    continue
            else:
                # 如果所有編碼嘗試都失敗，使用默認值
                sep = ','
                encoding = 'utf-8'
            
            # 檢查文件大小，對於小文件直接讀取
            file_size = file_path.stat().st_size
            if file_size < 100 * 1024 * 1024:  # 小於 100MB
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, 
                                  on_bad_lines='skip', low_memory=False)
                    logger.info(f"使用 {encoding} 編碼成功讀取 {file_path}，大小: {file_size/1024/1024:.2f} MB")
                    return df
                except Exception as e:
                    logger.warning(f"直接讀取文件失敗: {e}，嘗試分塊讀取")
                
            # 大文件分塊讀取
            logger.info(f"文件較大 ({file_size/1024/1024:.2f} MB)，使用分塊讀取")
            chunks = []
            chunk_count = 0
            total_rows = 0
            
            # 使用 ThreadPoolExecutor 並行處理數據塊
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_chunk = {}
                
                for chunk in pd.read_csv(file_path, encoding=encoding, sep=sep,
                                        on_bad_lines='skip', chunksize=chunksize, 
                                        low_memory=False):
                    total_rows += len(chunk)
                    future = executor.submit(lambda c: c.copy(), chunk)
                    future_to_chunk[future] = chunk_count
                    chunk_count += 1
                    
                    if chunk_count % 10 == 0:
                        logger.info(f"已讀取 {chunk_count} 個數據塊，共 {total_rows} 行")
                
                # 收集處理結果
                for future in future_to_chunk:
                    try:
                        chunks.append(future.result())
                    except Exception as e:
                        logger.warning(f"處理數據塊時出錯: {e}")
                        
            if not chunks:
                logger.warning(f"使用 {encoding} 分塊讀取無結果")
                return None
                    
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"分塊讀取完成，共 {len(df)} 行")
            return df
                
        elif ext in ['.xls', '.xlsx']:
            # 嘗試不同的 Excel 引擎
            engines = ['openpyxl', 'xlrd']
            for engine in engines:
                try:
                    df = pd.read_excel(file_path, engine=engine)
                    logger.info(f"使用 {engine} 引擎成功讀取 {file_path}")
                    return df
                except Exception as e:
                    logger.warning(f"使用 {engine} 引擎讀取 {file_path} 時出錯: {e}")
                    continue
            
            # 嘗試顯式指定 sheet_name
            try:
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                if sheet_names:
                    sheet_name = sheet_names[0]
                    logger.info(f"嘗試讀取工作表: {sheet_name}")
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    return df
            except Exception as e:
                logger.warning(f"嘗試讀取第一個工作表時出錯: {e}")
            
            logger.error(f"使用所有 Excel 引擎讀取 {file_path} 均失敗")
            return None
        else:
            logger.error(f"不支持的文件格式: {ext}")
            return None
    except Exception as e:
        logger.error(f"讀取 {file_path} 時發生嚴重錯誤: {e}")
        logger.error(traceback.format_exc())
        return None

def convert_units(df):
    """
    自動檢測並轉換數據單位，確保物理量的一致性。
    
    Args:
        df: 數據框
        
    Returns:
        單位轉換後的數據框
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # 根據您的數據結構，電壓和電流可能是整數，需要轉換
    # 電壓轉換 (如果是毫伏，需要轉為伏特)
    if 'voltage' in df.columns:
        # 檢查電壓範圍來決定是否需要轉換
        max_voltage = df['voltage'].max()
        if max_voltage > 100:  # 可能是毫伏單位
            logger.info(f"將電壓從 mV 轉換為 V (最大值: {max_voltage})")
            df['voltage'] = df['voltage'] * 0.001
    
    # 電流轉換 (如果是毫安，需要轉為安培)
    if 'current' in df.columns:
        try:
            # 使用絕對值來檢查範圍
            max_current_abs = abs(df['current']).max()
            if max_current_abs > 100:  # 可能是毫安單位
                logger.info(f"將電流從 mA 轉換為 A (最大絕對值: {max_current_abs})")
                df['current'] = df['current'] * 0.001
        except Exception as e:
            logger.warning(f"處理電流列時出錯: {e}")
            
    # 單元電壓轉換 (BMS_CellVolt1-13)
    for i in range(1, 14):
        cell_col = f'bms_cellvolt{i}'
        if cell_col in df.columns:
            try:
                max_cell_voltage = df[cell_col].max()
                if max_cell_voltage > 100:  # 可能是毫伏單位
                    logger.info(f"將 {cell_col} 從 mV 轉換為 V (最大值: {max_cell_voltage})")
                    df[cell_col] = df[cell_col] * 0.001
            except Exception as e:
                logger.warning(f"處理 {cell_col} 時出錯: {e}")
    
    # 其他可能需要轉換的列
    conversion_map = {
        'bms_rc': {'threshold': 100, 'factor': 0.001, 'from': 'mAh', 'to': 'Ah'},
        'bms_fcc': {'threshold': 100, 'factor': 0.001, 'from': 'mAh', 'to': 'Ah'},
        'bms_dcr': {'threshold': 1000, 'factor': 1e-6, 'from': 'μΩ', 'to': 'Ω'},
        'bms_fdcr': {'threshold': 1000, 'factor': 1e-6, 'from': 'μΩ', 'to': 'Ω'},
        'bms_avgcurrent': {'threshold': 100, 'factor': 0.001, 'from': 'mA', 'to': 'A'},
        'bms_chargingcurrent': {'threshold': 100, 'factor': 0.001, 'from': 'mA', 'to': 'A'},
    }
    
    # 處理其他可能需要轉換的列
    for col, info in conversion_map.items():
        if col in df.columns:
            try:
                # 對於電流相關列使用絕對值檢查
                if 'current' in col:
                    max_val = abs(df[col]).max()
                else:
                    max_val = df[col].max()
                    
                if max_val > info['threshold']:
                    logger.info(f"將 {col} 從 {info['from']} 轉換為 {info['to']} (最大值: {max_val})")
                    df[col] = df[col] * info['factor']
            except Exception as e:
                logger.warning(f"處理 {col} 時出錯: {e}")
    
    return df

class DataCleaningPipeline:
    """數據清理管道，提供流式的數據清理功能"""
    
    def __init__(self, data, verbose=False):
        """初始化數據清理管道
        
        Args:
            data: 輸入數據框，必須是 pd.DataFrame
            verbose: 是否詳細輸出
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("輸入數據必須是 pandas DataFrame")
        self.data = data.copy()
        self.original_rows = len(data)
        self.original_cols = len(data.columns)
        self.verbose = verbose
        self.steps_applied = []
        self.metrics = {}
    
    def add_metric(self, name: str, value: Any) -> 'DataCleaningPipeline':
        """添加清理指標
        
        Args:
            name: 指標名稱
            value: 指標值
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        self.metrics[name] = value
        return self
    
    @log_step("預處理數據")
    def preprocess(self) -> 'DataCleaningPipeline':
        """
        數據預處理：轉換數據類型、處理極端異常值
        
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            # 檢查是否有大寫列名，如果有，統一轉為小寫
            uppercase_cols = [col for col in self.data.columns if any(c.isupper() for c in col)]
            if uppercase_cols and self.verbose:
                logger.info(f"檢測到 {len(uppercase_cols)} 個大寫列名，將統一轉為小寫")
                # 在unify_columns中處理，此處不實際轉換
            
            # 處理數據類型
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    if col.lower() in ['datetime', 'date', 'time', 'timestamp']:
                        # 時間列處理
                        try:
                            # 指定 DateTime 格式
                            self.data[col] = pd.to_datetime(self.data[col], errors='coerce', format='%Y/%m/%d %p %I:%M:%S')
                            if self.verbose:
                                logger.info(f"列 '{col}' 已轉換為日期時間型")
                        except:
                            pass
                    else:
                        # 非時間列嘗試轉為數值
                        try:
                            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                            if self.data[col].notna().any() and self.verbose:
                                logger.info(f"列 '{col}' 已轉換為數值型")
                        except:
                            continue
            
            # 移除完全空白的行和列
            na_rows_before = self.data.isna().all(axis=1).sum()
            na_cols_before = self.data.isna().all(axis=0).sum()
            
            if na_rows_before > 0:
                self.data = self.data.dropna(how='all')
                if self.verbose:
                    logger.info(f"移除了 {na_rows_before} 行全空白數據")
            
            if na_cols_before > 0:
                self.data = self.data.dropna(axis=1, how='all')
                if self.verbose:
                    logger.info(f"移除了 {na_cols_before} 列全空白數據")
            
            # 檢測並處理異常的巨大數值（基於物理約束）
            num_cols = self.data.select_dtypes(include=[np.number]).columns
            extreme_outliers_summary = {}
            
            for col in num_cols:
                try:
                    col_lower = col.lower()
                    matching_col = None
                    
                    # 嘗試匹配物理約束
                    for constraint_col, constraint in PHYSICAL_CONSTRAINTS.items():
                        if constraint_col.lower() == col_lower or col_lower == constraint_col.lower().replace('bms_', ''):
                            matching_col = constraint_col
                            break
                    
                    if matching_col:
                        constraint = PHYSICAL_CONSTRAINTS[matching_col]
                        # 考慮單位變換
                        min_val = constraint['min']
                        max_val = constraint['max']
                        
                        # 根據單位自動調整約束
                        if 'volt' in col_lower and min_val > 100 and self.data[col].max() < 10:
                            min_val /= 1000.0
                            max_val /= 1000.0
                        elif 'curr' in col_lower and min_val > 10 and abs(self.data[col]).max() < 10:
                            min_val /= 1000.0
                            max_val /= 1000.0
                        
                        outliers = ((self.data[col] < min_val) | 
                                  (self.data[col] > max_val)) & ~self.data[col].isna()
                        outlier_count = outliers.sum()
                        
                        if outlier_count > 0:
                            median_val = self.data[col].median()
                            self.data.loc[outliers, col] = median_val
                            extreme_outliers_summary[col] = int(outlier_count)
                            if self.verbose:
                                logger.warning(f"列 '{col}' 中檢測到 {outlier_count} 個極端異常值（超出範圍 {min_val}-{max_val}），已替換為中位數 {median_val}")
                except Exception as e:
                    logger.warning(f"處理列 '{col}' 的極端異常值時出錯: {e}")
                    continue
            
            if extreme_outliers_summary:
                self.add_metric("extreme_outliers_replaced", extreme_outliers_summary)
            
            self.steps_applied.append("preprocess")
        except Exception as e:
            logger.error(f"預處理數據時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
                
        return self
    
    
    @log_step("統一欄位名稱")
    def unify_columns(self) -> 'DataCleaningPipeline':
        """统一列名，使用标准别名映射，同时保留 DateTime 列"""
        try:
            original_cols = self.data.columns.tolist()
            if self.verbose:
                logger.info(f"原始列: {original_cols}")
            
            # Check for DateTime column
            has_datetime = 'DateTime' in original_cols
            
            # Create a copy of DateTime column if it exists
            if has_datetime:
                self.data['original_DateTime'] = self.data['DateTime']
                if self.verbose:
                    logger.info("创建 DateTime 列的备份")
            
            # 建立大小寫不敏感的映射表
            case_insensitive_map = {}
            for col in self.data.columns:
                col_lower = col.lower().strip()
                if col_lower in map(str.lower, COLUMN_ALIASES.keys()):
                    # 找到匹配的別名，不區分大小寫
                    for alias, standard in COLUMN_ALIASES.items():
                        if alias.lower() == col_lower:
                            case_insensitive_map[col] = standard
                            if self.verbose:
                                logger.info(f"匹配列名 {col} -> {standard} (大小寫不敏感)")
                            break
            
            # 應用映射
            self.data = self.data.rename(columns=case_insensitive_map)
            
            # Restore DateTime from backup if it existed
            if has_datetime and 'original_DateTime' in self.data.columns:
                self.data['DateTime'] = self.data['original_DateTime']
                self.data = self.data.drop(columns=['original_DateTime'])
                if self.verbose:
                    logger.info("恢复 DateTime 列")
            
            if self.verbose:
                logger.info(f"重命名后列: {self.data.columns.tolist()}")
                
            self.steps_applied.append("unify_columns")
        except Exception as e:
            logger.error(f"统一列名时出错: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
            
        return self
    
    @log_step("單位轉換")
    def convert_units(self) -> 'DataCleaningPipeline':
        """轉換單位以確保物理一致性"""
        try:
            # 檢查數據框是否為空
            if self.data is None or self.data.empty:
                logger.error("數據框為空，無法進行單位轉換")
                return self

            # 創建大小寫不敏感的列名映射
            col_mapping = {col.lower(): col for col in self.data.columns}
            
            # 轉換 BMS_PackVoltage/voltage 從 mV 到 V
            for voltage_col in ['bms_packvoltage', 'voltage']:
                if voltage_col in col_mapping:
                    col = col_mapping[voltage_col]
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        max_value = self.data[col].max()
                        if max_value > 100:  # 假設超過100是mV單位
                            self.data[col] = self.data[col] / 1000.0
                            logger.info(f"已將 {col} 從 mV 轉換為 V (最大值: {max_value})")
                    else:
                        logger.warning(f"{col} 列不是數值類型，跳過單位轉換")

            # 轉換 BMS_PackCurrent/current 從 mA 到 A
            for current_col in ['bms_packcurrent', 'current', 'bms_avgcurrent']:
                if current_col in col_mapping:
                    col = col_mapping[current_col]
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        max_abs = abs(self.data[col]).max()
                        if max_abs > 100:  # 假設超過100是mA單位
                            self.data[col] = self.data[col] / 1000.0
                            logger.info(f"已將 {col} 從 mA 轉換為 A (最大絕對值: {max_abs})")
                    else:
                        logger.warning(f"{col} 列不是數值類型，跳過單位轉換")
            
            # 單元電壓轉換
            for i in range(1, 14):
                cell_col_options = [f'bms_cellvolt{i}', f'BMS_CellVolt{i}', f'cellvolt{i}']
                for option in cell_col_options:
                    if option.lower() in col_mapping:
                        col = col_mapping[option.lower()]
                        if pd.api.types.is_numeric_dtype(self.data[col]):
                            max_value = self.data[col].max()
                            if max_value > 10:  # 單元電壓超過10V不合理
                                # 可能是mV單位
                                self.data[col] = self.data[col] / 1000.0
                                logger.info(f"已將 {col} 從 mV 轉換為 V (最大值: {max_value})")
                        break  # 找到一個有效列後就不需要檢查其它選項

            # 記錄操作步驟
            self.steps_applied.append("convert_units")
            logger.info("單位轉換步驟已完成")

        except Exception as e:
            logger.error(f"單位轉換失敗: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())

        return self
    
    @log_step("處理時間戳")
    def handle_timestamps(self) -> 'DataCleaningPipeline':
        """处理时间戳，包括解析和修复，同时保留 DateTime 列"""
        if 'time' not in self.data.columns and 'DateTime' not in self.data.columns:
            if self.verbose:
                logger.warning("缺少时间列，跳过时间戳处理")
            return self
                
        try:
            # Parse datetime for the 'time' column
            if 'time' in self.data.columns:
                self.data = robust_datetime_parsing(self.data, datetime_col='time')
                
            # Also process DateTime if it exists
            if 'DateTime' in self.data.columns:
                datetime_df = robust_datetime_parsing(self.data, datetime_col='DateTime')
                # Update only the parsed_dt column from datetime_df
                if 'parsed_dt' in datetime_df.columns:
                    self.data['parsed_dt'] = datetime_df['parsed_dt']
                
            # Fix timestamps
            processed_df = fix_timestamps(self.data, dt_col='parsed_dt', out_col='timeindex')
            
            # Only update if processed data is not empty
            if not processed_df.empty:
                original_rows = len(self.data)
                
                # Ensure DateTime is preserved in the processed dataframe
                if 'DateTime' in self.data.columns and 'DateTime' not in processed_df.columns:
                    processed_df['DateTime'] = self.data.loc[processed_df.index, 'DateTime'].values
                    
                self.data = processed_df
                filtered_rows = original_rows - len(self.data)
                
                if self.verbose and filtered_rows > 0:
                    logger.info(f"时间戳修复过滤掉了 {filtered_rows} 行数据")
                
                self.add_metric("timestamp_filtered_rows", filtered_rows)
            
            self.steps_applied.append("handle_timestamps")
        except Exception as e:
            logger.error(f"处理时间戳时出错: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("標記充電與放電")
    def mark_charge_discharge(self, current_col='current', threshold=0) -> 'DataCleaningPipeline':
        """標記數據為充電和放電階段
        
        Args:
            current_col: 電流列名
            threshold: 區分充放電的閾值
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            if current_col not in self.data.columns:
                # 嘗試找到匹配的電流列
                current_cols = [col for col in self.data.columns 
                              if 'current' in col.lower() and not 'charging' in col.lower()]
                if current_cols:
                    current_col = current_cols[0]
                    logger.info(f"使用 {current_col} 作為電流列")
                else:
                    logger.warning(f"找不到電流列，無法標記充放電階段")
                    return self
            
            # 確保電流列是數值型
            if not pd.api.types.is_numeric_dtype(self.data[current_col]):
                try:
                    self.data[current_col] = pd.to_numeric(self.data[current_col], errors='coerce')
                    logger.info(f"已將 {current_col} 轉換為數值型")
                except Exception as e:
                    logger.warning(f"轉換 {current_col} 為數值型時出錯: {e}")
                    return self
            
            # 標記充電和放電
            self.data['is_charging'] = self.data[current_col] <= threshold
            
            # 計算充電和放電數據的比例
            charging_count = self.data['is_charging'].sum()
            discharging_count = len(self.data) - charging_count
            charging_pct = charging_count / len(self.data) * 100 if len(self.data) > 0 else 0
            
            if self.verbose:
                logger.info(f"數據標記完成: 充電階段 {charging_count} 行 ({charging_pct:.2f}%)，"
                           f"放電階段 {discharging_count} 行 ({100-charging_pct:.2f}%)")
            
            self.add_metric("charging_data", {
                "count": int(charging_count),
                "percentage": float(charging_pct)
            })
            
            self.add_metric("discharging_data", {
                "count": int(discharging_count),
                "percentage": float(100-charging_pct)
            })
            
            self.steps_applied.append("mark_charge_discharge")
        except Exception as e:
            logger.error(f"標記充放電階段時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("過濾充電數據")
    def filter_charging_data(self, keep_charging=False) -> 'DataCleaningPipeline':
        """過濾充電階段數據，只保留放電數據
        
        Args:
            keep_charging: 是否保留充電數據
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            if 'is_charging' not in self.data.columns:
                logger.warning("數據未標記充放電階段，先執行標記")
                self = self.mark_charge_discharge()
                if 'is_charging' not in self.data.columns:
                    logger.warning("無法標記充放電階段，跳過過濾")
                    return self
            
            original_count = len(self.data)
            
            if not keep_charging:
                # 只保留放電數據
                self.data = self.data[~self.data['is_charging']].reset_index(drop=True)
                filtered_count = original_count - len(self.data)
                
                if self.verbose:
                    logger.info(f"已過濾掉 {filtered_count} 行充電數據 ({filtered_count/original_count*100:.2f}%)")
                
                self.add_metric("charging_data_filtered", {
                    "original_count": int(original_count),
                    "filtered_count": int(filtered_count),
                    "percentage": float(filtered_count/original_count*100)
                })
                
                self.steps_applied.append("filter_charging_data")
            else:
                if self.verbose:
                    logger.info("保留充電數據")
        except Exception as e:
            logger.error(f"過濾充電數據時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("檢查缺失值")
    def check_missing_values(self) -> 'DataCleaningPipeline':
        """檢查並報告缺失值
        
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            missing_counts = self.data.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            
            if len(missing_cols) > 0:
                missing_percentage = (missing_counts / len(self.data) * 100).round(2)
                missing_info = {}
                
                if self.verbose:
                    logger.warning("發現缺失值:")
                    
                for col, count in missing_cols.items():
                    percentage = missing_percentage[col]
                    missing_info[col] = {"count": int(count), "percentage": float(percentage)}
                    
                    if self.verbose:
                        logger.warning(f"  {col}: {count} 缺失值 ({percentage:.2f}%)")
                
                self.add_metric("missing_values", missing_info)
                
                # 檢查嚴重缺失的列 (>80%)
                severe_missing = [col for col, info in missing_info.items() 
                                 if info["percentage"] > 80]
                if severe_missing and self.verbose:
                    logger.warning(f"以下列的缺失值超過80%，可能需要特別處理: {severe_missing}")
                    self.add_metric("severe_missing_columns", severe_missing)
            else:
                if self.verbose:
                    logger.info("數據中沒有缺失值")
            
        except Exception as e:
            logger.error(f"檢查缺失值時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("插補缺失值")
    def interpolate_missing(self, method: str = 'linear') -> 'DataCleaningPipeline':
        """插補缺失值
        
        Args:
            method: 插補方法，默認為線性插補
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            # 先將所有對象類型列先排除
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            string_cols = self.data.select_dtypes(include=['object']).columns
            datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
            
            # 排除時間索引列，它們通常不需要插補
            numeric_cols = [col for col in numeric_cols if col not in ['timeindex']]
            
            # 計算數值列的缺失值數量
            na_before = self.data[numeric_cols].isna().sum().sum()
            
            interpolation_summary = {}
            
            if na_before > 0:
                if self.verbose:
                    logger.info(f"使用 {method} 方法插補 {na_before} 個數值型缺失值")
                
                # 對數值列執行插補
                self.data[numeric_cols] = self.data[numeric_cols].interpolate(method=method, limit_direction='both')
                
                # 如果仍有缺失值，使用前向和後向填充
                remaining_na = self.data[numeric_cols].isna().sum().sum()
                if remaining_na > 0:
                    self.data[numeric_cols] = self.data[numeric_cols].fillna(method='ffill')
                    self.data[numeric_cols] = self.data[numeric_cols].fillna(method='bfill')
                    
                    if self.verbose:
                        logger.info(f"插補後仍有 {remaining_na} 個缺失值，使用前向和後向填充方法處理")
                
                # 如果仍有缺失值，使用均值填充
                final_na = self.data[numeric_cols].isna().sum().sum()
                if final_na > 0:
                    if self.verbose:
                        logger.warning(f"插補後仍有 {final_na} 個數值型缺失值，使用均值填充")
                    
                    for col in numeric_cols:
                        if self.data[col].isna().any():
                            mean_val = self.data[col].mean()
                            if pd.isna(mean_val):  # 如果均值也是 NaN
                                mean_val = 0
                            self.data[col] = self.data[col].fillna(mean_val)
                            if self.verbose:
                                logger.info(f"列 '{col}' 使用均值 {mean_val} 填充")
                
                interpolation_summary["numeric"] = {
                    "before": int(na_before),
                    "after_interpolation": int(remaining_na),
                    "final": int(self.data[numeric_cols].isna().sum().sum())
                }
            
            # 處理日期時間列的缺失值
            if len(datetime_cols) > 0:
                dt_na_before = self.data[datetime_cols].isna().sum().sum()
                if dt_na_before > 0:
                    if self.verbose:
                        logger.info(f"使用前向和後向填充處理 {dt_na_before} 個日期時間型缺失值")
                    
                    # 使用前向和後向填充
                    self.data[datetime_cols] = self.data[datetime_cols].fillna(method='ffill')
                    self.data[datetime_cols] = self.data[datetime_cols].fillna(method='bfill')
                    
                    interpolation_summary["datetime"] = {
                        "before": int(dt_na_before),
                        "final": int(self.data[datetime_cols].isna().sum().sum())
                    }
            
            # 處理字符串列的缺失值
            if len(string_cols) > 0:
                str_na_before = self.data[string_cols].isna().sum().sum()
                if str_na_before > 0:
                    if self.verbose:
                        logger.info(f"使用前向填充處理 {str_na_before} 個字符串型缺失值")
                    
                    # 使用前向和後向填充
                    self.data[string_cols] = self.data[string_cols].fillna(method='ffill')
                    self.data[string_cols] = self.data[string_cols].fillna(method='bfill')
                    
                    # 如果仍有缺失值，使用空字符串填充
                    str_na_after = self.data[string_cols].isna().sum().sum()
                    if str_na_after > 0:
                        self.data[string_cols] = self.data[string_cols].fillna('')
                        if self.verbose:
                            logger.info(f"仍有 {str_na_after} 個字符串型缺失值，使用空字符串填充")
                    
                    interpolation_summary["string"] = {
                        "before": int(str_na_before),
                        "final": int(self.data[string_cols].isna().sum().sum())
                    }
            
            if interpolation_summary:
                self.add_metric("interpolation_summary", interpolation_summary)
            
            self.steps_applied.append("interpolate_missing")
        except Exception as e:
            logger.error(f"插補缺失值時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    def detrend_time_series(self, columns: List[str], window: int = 60) -> 'DataCleaningPipeline':
        """對指定列進行趨勢校正，計算與移動平均的偏差"""
        try:
            # 檢查列是否存在，只處理存在的列
            existing_columns = []
            for col in columns:
                if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col]):
                    existing_columns.append(col)
                else:
                    logger.warning(f"列 '{col}' 不存在或不是數值類型，跳過趨勢校正")
            
            for col in existing_columns:
                # 計算移動平均
                moving_avg = self.data[col].rolling(window=window, min_periods=1, center=True).mean()
                # 計算偏差
                self.data[f'{col}_detrended'] = self.data[col] - moving_avg
                logger.info(f"已對列 '{col}' 進行趨勢校正（窗口大小={window}）")
        except Exception as e:
            logger.error(f"趨勢校正時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        return self
    
    @log_step("單位轉換")
    def convert_units(self) -> 'DataCleaningPipeline':
        """轉換單位以確保物理一致性

        具體轉換：
        - BMS_PackVoltage: 從 mV 轉換為 V（除以 1000）
        - BMS_PackCurrent: 從 mA 轉換為 A（除以 1000）
        - BMS_AvgCurrent: 從 mA 轉換為 A（除以 1000）

        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            # 檢查數據框是否為空
            if self.data is None or self.data.empty:
                logger.error("數據框為空，無法進行單位轉換")
                return self

            # 轉換 BMS_PackVoltage 從 mV 到 V
            if 'BMS_PackVoltage' in self.data.columns:
                if pd.api.types.is_numeric_dtype(self.data['BMS_PackVoltage']):
                    self.data['BMS_PackVoltage'] = self.data['BMS_PackVoltage'] / 1000.0
                    logger.info("已將 BMS_PackVoltage 從 mV 轉換為 V")
                else:
                    logger.warning("BMS_PackVoltage 列不是數值類型，跳過單位轉換")

            # 轉換 BMS_PackCurrent 從 mA 到 A
            if 'BMS_PackCurrent' in self.data.columns:
                if pd.api.types.is_numeric_dtype(self.data['BMS_PackCurrent']):
                    self.data['BMS_PackCurrent'] = self.data['BMS_PackCurrent'] / 1000.0
                    logger.info("已將 BMS_PackCurrent 從 mA 轉換為 A")
                else:
                    logger.warning("BMS_PackCurrent 列不是數值類型，跳過單位轉換")

            # 轉換 BMS_AvgCurrent 從 mA 到 A
            if 'BMS_AvgCurrent' in self.data.columns:
                if pd.api.types.is_numeric_dtype(self.data['BMS_AvgCurrent']):
                    self.data['BMS_AvgCurrent'] = self.data['BMS_AvgCurrent'] / 1000.0
                    logger.info("已將 BMS_AvgCurrent 從 mA 轉換為 A")
                else:
                    logger.warning("BMS_AvgCurrent 列不是數值類型，跳過單位轉換")

            # 記錄操作步驟
            self.steps_applied.append("convert_units")
            logger.info("單位轉換步驟已完成")

        except Exception as e:
            logger.error(f"單位轉換失敗: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())

        return self
    
    @log_step("處理異常值")
    def handle_outliers(self, columns: Optional[List[str]] = None, method: str = 'iqr', 
                    threshold: float = 1.5, column_thresholds: Optional[Dict[str, float]] = None,
                    respect_charge_discharge: bool = True) -> 'DataCleaningPipeline':
        """處理異常值
        
        Args:
            columns: 要處理的列，為None則處理所有數值列
            method: 異常值檢測方法，'iqr'或'zscore'
            threshold: 默認閾值，IQR方法為幾倍IQR，Z分數方法為幾個標準差
            column_thresholds: 特定列的閾值映射，覆蓋默認閾值
            respect_charge_discharge: 是否考慮充放電狀態分別處理異常值
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            # 如果未指定列，處理所有數值列
            if columns is None:
                columns = self.data.select_dtypes(include=[np.number]).columns
            else:
                # 只保留存在的列
                columns = [col for col in columns if col in self.data.columns]
            
            # 設置 column_thresholds 為空字典如果未提供
            if column_thresholds is None:
                column_thresholds = {}
            
            outlier_counts = {}
            total_fixed = 0
            
            # 檢查是否有充放電標記
            has_charge_discharge = 'is_charging' in self.data.columns and respect_charge_discharge
            
            # 處理方法
            def process_outliers(data, col, col_threshold, method_name):
                """處理單一列的異常值"""
                if method_name == 'iqr':
                    # IQR方法
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - col_threshold * IQR
                    upper_bound = Q3 + col_threshold * IQR
                elif method_name == 'zscore':
                    # Z分數方法
                    mean = data[col].mean()
                    std = data[col].std()
                    
                    # 防止標準差為0的情況
                    if std == 0:
                        return data, 0, None, None
                    
                    lower_bound = mean - col_threshold * std
                    upper_bound = mean + col_threshold * std
                else:
                    logger.warning(f"未知的異常值檢測方法: {method_name}")
                    return data, 0, None, None
                
                # 標記異常值
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                # 過濾缺失值
                outliers = outliers & ~data[col].isna()
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # 複製數據以避免修改原始數據
                    data_copy = data.copy()
                    
                    # 用邊界值替換異常值
                    data_copy.loc[data_copy[col] < lower_bound, col] = lower_bound
                    data_copy.loc[data_copy[col] > upper_bound, col] = upper_bound
                    
                    return data_copy, outlier_count, lower_bound, upper_bound
                else:
                    return data, 0, lower_bound, upper_bound
            
            for col in columns:
                # 跳過時間或標識符列
                if col in ['time', 'timeindex', 'parsed_dt', 'is_charging']:
                    continue
                    
                # 確保列是數值型
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    if self.verbose:
                        logger.warning(f"列 '{col}' 不是數值型，跳過異常值處理")
                    continue
                    
                # 確保不全是缺失值
                if self.data[col].isna().all():
                    if self.verbose:
                        logger.warning(f"列 '{col}' 全為缺失值，跳過異常值處理")
                    continue
                
                try:
                    # 使用特定列的閾值，如果未指定則使用默認閾值
                    col_threshold = column_thresholds.get(col, threshold)
                    
                    if has_charge_discharge:
                        # 分別處理充電和放電數據
                        charging_data = self.data[self.data['is_charging']]
                        discharging_data = self.data[~self.data['is_charging']]
                        
                        charge_outliers = 0
                        discharge_outliers = 0
                        
                        if not charging_data.empty:
                            charging_data, charge_outliers, c_lower, c_upper = process_outliers(
                                charging_data, col, col_threshold, method
                            )
                        
                        if not discharging_data.empty:
                            discharging_data, discharge_outliers, d_lower, d_upper = process_outliers(
                                discharging_data, col, col_threshold, method
                            )
                        
                        if charge_outliers > 0 or discharge_outliers > 0:
                            # 合併處理後的數據
                            self.data = pd.concat([charging_data, discharging_data]).sort_index()
                            
                            # 記錄統計信息
                            outlier_count = charge_outliers + discharge_outliers
                            outlier_percent = outlier_count / len(self.data) * 100
                            
                            if self.verbose:
                                logger.info(f"列 '{col}' 中檢測到 {outlier_count} 個異常值 ({outlier_percent:.2f}%)，"
                                          f"充電: {charge_outliers}，放電: {discharge_outliers}")
                            
                            outlier_counts[col] = {
                                "count": int(outlier_count),
                                "percentage": float(outlier_percent),
                                "charge_count": int(charge_outliers),
                                "discharge_count": int(discharge_outliers),
                                "threshold": float(col_threshold)
                            }
                            
                            total_fixed += outlier_count
                            
                    else:
                        # 常規處理（不區分充放電）
                        processed_data, outlier_count, lower_bound, upper_bound = process_outliers(
                            self.data, col, col_threshold, method
                        )
                        
                        if outlier_count > 0:
                            self.data = processed_data
                            
                            # 計算異常值百分比
                            outlier_percent = outlier_count / len(self.data) * 100
                            
                            if self.verbose:
                                logger.info(f"列 '{col}' 中檢測到 {outlier_count} 個異常值 ({outlier_percent:.2f}%)，"
                                           f"使用閾值 {col_threshold}")
                            
                            outlier_counts[col] = {
                                "count": int(outlier_count),
                                "percentage": float(outlier_percent),
                                "lower_bound": float(lower_bound),
                                "upper_bound": float(upper_bound),
                                "threshold": float(col_threshold)
                            }
                            
                            total_fixed += outlier_count
                        
                except Exception as e:
                    logger.warning(f"處理列 '{col}' 的異常值時出錯: {e}")
                    if self.verbose:
                        logger.error(traceback.format_exc())
                    continue
            
            if outlier_counts:
                if self.verbose:
                    logger.info(f"共處理了 {total_fixed} 個異常值")
                    
                self.add_metric("outliers_replaced", outlier_counts)
                self.add_metric("total_outliers_fixed", int(total_fixed))
                self.steps_applied.append("handle_outliers")
            else:
                if self.verbose:
                    logger.info("未檢測到需要處理的異常值")
        
        except Exception as e:
            logger.error(f"處理異常值時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("檢查物理約束")
    def check_physical_constraints(self) -> 'DataCleaningPipeline':
        """檢查數據是否滿足物理約束
        
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            constraints_fixed = {}
            total_fixed = 0
            
            # 創建大小寫不敏感的列名映射
            col_mapping = {col.lower(): col for col in self.data.columns}
            
            for col, constraint in PHYSICAL_CONSTRAINTS.items():
                # 大小寫不敏感匹配
                matching_col = None
                if col.lower() in col_mapping:
                    matching_col = col_mapping[col.lower()]
                elif col in self.data.columns:
                    matching_col = col
                
                if matching_col and pd.api.types.is_numeric_dtype(self.data[matching_col]):
                    # 檢查單位，視需要轉換約束條件
                    # 例如，如果電壓列已經轉換為V，但約束仍然是mV範圍
                    col_max = self.data[matching_col].max()
                    min_constraint = constraint['min']
                    max_constraint = constraint['max']
                    
                    # 檢查單位可能不匹配的情況
                    if 'volt' in matching_col.lower() and col_max < 10 and constraint['max'] > 1000:
                        # 資料已是V，但約束是mV
                        min_constraint /= 1000.0
                        max_constraint /= 1000.0
                        if self.verbose:
                            logger.info(f"自動調整 {matching_col} 的約束單位從 mV 到 V")
                    elif 'curr' in matching_col.lower() and col_max < 10 and constraint['max'] > 1000:
                        # 資料已是A，但約束是mA
                        min_constraint /= 1000.0
                        max_constraint /= 1000.0
                        if self.verbose:
                            logger.info(f"自動調整 {matching_col} 的約束單位從 mA 到 A")
                    
                    # 過濾缺失值
                    outliers = ((self.data[matching_col] < min_constraint) | 
                            (self.data[matching_col] > max_constraint)) & ~self.data[matching_col].isna()
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        if self.verbose:
                            logger.info(f"修正 {outlier_count} 個超出範圍的 {matching_col} 值 "
                                    f"(限制: {min_constraint}-{max_constraint} {constraint['unit']})")
                        
                        # 使用中位數或者邊界值替換異常值
                        self.data.loc[self.data[matching_col] < min_constraint, matching_col] = min_constraint
                        self.data.loc[self.data[matching_col] > max_constraint, matching_col] = max_constraint
                        
                        constraints_fixed[matching_col] = {
                            "count": int(outlier_count),
                            "min": float(min_constraint),
                            "max": float(max_constraint),
                            "unit": constraint['unit']
                        }
                        total_fixed += outlier_count
            
            # 處理電池單元電壓 (使用現有的列名)
            cell_fixed = 0
            # 找到所有包含 cellvolt 的列
            cell_cols = [col for col in self.data.columns if 'cellvolt' in col.lower()]
            for cell_col in cell_cols:
                if pd.api.types.is_numeric_dtype(self.data[cell_col]):
                    # 檢查資料範圍判斷單位
                    col_max = self.data[cell_col].max()
                    min_limit = 2.5  # V
                    max_limit = 4.5  # V
                    
                    # 調整閾值如果數據是mV
                    if col_max > 1000:
                        min_limit = 2500  # mV
                        max_limit = 4500  # mV
                    
                    cell_outliers = ((self.data[cell_col] < min_limit) | 
                                (self.data[cell_col] > max_limit)) & ~self.data[cell_col].isna()
                    cell_count = cell_outliers.sum()
                    
                    if cell_count > 0:
                        if self.verbose:
                            logger.info(f"修正 {cell_count} 個超出範圍的 {cell_col} 值 (限制: {min_limit}-{max_limit})")
                        
                        self.data.loc[self.data[cell_col] < min_limit, cell_col] = min_limit
                        self.data.loc[self.data[cell_col] > max_limit, cell_col] = max_limit
                        
                        unit = "mV" if min_limit > 1000 else "V"
                        constraints_fixed[cell_col] = {
                            "count": int(cell_count),
                            "min": float(min_limit),
                            "max": float(max_limit),
                            "unit": unit
                        }
                        cell_fixed += cell_count
            
            if constraints_fixed:
                if self.verbose:
                    logger.info(f"共修正了 {total_fixed + cell_fixed} 個超出物理約束的值")
                    
                self.add_metric("constraints_fixed", constraints_fixed)
                self.add_metric("total_constraints_fixed", int(total_fixed + cell_fixed))
                self.steps_applied.append("check_physical_constraints")
            else:
                if self.verbose:
                    logger.info("未檢測到超出物理約束的值")
            
        except Exception as e:
            logger.error(f"檢查物理約束時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
            
        return self
    
    @log_step("修復時間序列")
    def fix_time_series(self, time_column: str = 'time') -> 'DataCleaningPipeline':
        """修復時間序列，確保時間是單調遞增的
        
        Args:
            time_column: 時間列名
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            if time_column not in self.data.columns:
                if self.verbose:
                    logger.warning(f"時間列 '{time_column}' 不存在，跳過時間序列修復")
                return self
            
            # 檢查時間列類型
            if not pd.api.types.is_datetime64_any_dtype(self.data[time_column]):
                try:
                    self.data[time_column] = pd.to_datetime(self.data[time_column], errors='coerce')
                    if self.verbose:
                        logger.info(f"已將時間列 '{time_column}' 轉換為日期時間型")
                except Exception as e:
                    logger.warning(f"無法將列 '{time_column}' 轉換為日期時間: {e}")
                    return self
            
            # 檢查缺失值
            na_count = self.data[time_column].isna().sum()
            if na_count > 0:
                if self.verbose:
                    logger.warning(f"時間列 '{time_column}' 中有 {na_count} 個缺失值，將嘗試填充")
                self.data = self.data.dropna(subset=[time_column])
                if self.data.empty:
                    logger.warning(f"時間列 '{time_column}' 全為缺失值，無法修復時間序列")
                    return self
            
            # 檢查時間序列是否有重複值
            duplicated_times = self.data.duplicated(subset=[time_column], keep=False)
            duplicated_count = duplicated_times.sum()
            
            if duplicated_count > 0:
                if self.verbose:
                    logger.warning(f"發現 {duplicated_count} 個重複的時間戳，保留最後一個")
                
                # 以更精確的方式處理重複的時間戳
                self.data = self.data.sort_values(time_column).drop_duplicates(subset=[time_column], keep='last')
                self.add_metric("duplicated_times_removed", int(duplicated_count))
            
            # 排序數據
            self.data = self.data.sort_values(time_column).reset_index(drop=True)
            
            # 檢查時間間隔
            time_diffs = self.data[time_column].diff().dropna()
            negative_diffs = (time_diffs < pd.Timedelta(0)).sum()
            
            if negative_diffs > 0:
                if self.verbose:
                    logger.warning(f"發現 {negative_diffs} 個時間倒退情況")
                
                self.add_metric("negative_time_diffs", int(negative_diffs))
            
            # 檢查異常大的時間間隔
            time_diff_seconds = time_diffs.dt.total_seconds()
            mean_interval = time_diff_seconds.median()  # 使用中位數作為參考
            
            if mean_interval > 0:
                # 查找超過均值10倍的間隔
                large_gaps = time_diff_seconds[time_diff_seconds > mean_interval * 10]
                if not large_gaps.empty:
                    gap_info = {
                        "count": len(large_gaps),
                        "max_gap_seconds": float(large_gaps.max()),
                        "mean_interval": float(mean_interval)
                    }
                    if self.verbose:
                        logger.warning(f"發現 {gap_info['count']} 個異常大的時間間隔，最大間隔為 {gap_info['max_gap_seconds']:.2f} 秒")
                    self.add_metric("time_series_gaps", gap_info)
            
            self.steps_applied.append("fix_time_series")
        
        except Exception as e:
            logger.error(f"修復時間序列時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("轉換數據類型")
    def convert_dtypes(self) -> 'DataCleaningPipeline':
        """轉換數據類型，優化記憶體使用
        
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            memory_before = self.data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            
            for col in self.data.columns:
                # 跳過特定列
                if col in ['time', 'parsed_dt']:
                    continue
                    
                # 嘗試將數值列轉換為float32
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    try:
                        # 檢查數值範圍是否適合float32
                        min_val = self.data[col].min()
                        max_val = self.data[col].max()
                        
                        # 如果數值範圍適合，轉換為float32
                        if (min_val > -3.4e38 and max_val < 3.4e38) or pd.isna(min_val) or pd.isna(max_val):
                            self.data[col] = self.data[col].astype(np.float32)
                        # 否則保持原樣
                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"無法將列 '{col}' 轉換為float32: {e}")
            
            memory_after = self.data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            memory_saved = memory_before - memory_after
            
            if self.verbose:
                logger.info(f"數據類型轉換: 從 {memory_before:.2f} MB 優化至 {memory_after:.2f} MB，節省了 {memory_saved:.2f} MB ({memory_saved/memory_before*100:.2f}%)")
            
            self.add_metric("memory_optimization", {
                "before_mb": float(memory_before),
                "after_mb": float(memory_after),
                "saved_mb": float(memory_saved),
                "saved_percent": float(memory_saved/memory_before*100)
            })
            
            self.steps_applied.append("convert_dtypes")
        
        except Exception as e:
            logger.error(f"轉換數據類型時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("過濾溫度")
    def filter_by_temperature(self, temp_filter: str) -> 'DataCleaningPipeline':
        """按溫度過濾數據
        
        Args:
            temp_filter: 溫度標籤，如"25deg"
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            # 首先檢查是否有溫度相關列
            temp_columns = [col for col in self.data.columns 
                            if 'temp' in col.lower() and 'mos' not in col.lower()]
            
            # 如果找不到'temp'列但有其他溫度列，使用第一個溫度列
            if 'temp' not in self.data.columns and temp_columns:
                self.data['temp'] = self.data[temp_columns[0]]
                if self.verbose:
                    logger.info(f"創建了'temp'列，使用現有的 {temp_columns[0]} 列")
            
            if 'temp' not in self.data.columns:
                # 檢查是否有 BMS_Temp1 (從您的數據來看，這是主要溫度列)
                if 'BMS_Temp1' in self.data.columns:
                    self.data['temp'] = self.data['BMS_Temp1']
                    if self.verbose:
                        logger.info(f"創建了'temp'列，使用 BMS_Temp1 列")
                else:
                    if self.verbose:
                        logger.warning("數據中沒有任何溫度相關列，無法進行溫度過濾")
                    return self
            
            # 確保temp列是數值型
            if not pd.api.types.is_numeric_dtype(self.data['temp']):
                try:
                    self.data['temp'] = pd.to_numeric(self.data['temp'], errors='coerce')
                    if self.verbose:
                        logger.info("已將 'temp' 列轉換為數值型")
                except Exception as e:
                    logger.warning(f"無法將 'temp' 列轉換為數值: {e}")
                    return self
            
            # 溫度匹配
            if temp_filter == '5deg':
                temp_range = (0, 10)
            elif temp_filter == '25deg':
                temp_range = (20, 30)
            elif temp_filter == '45deg':
                temp_range = (40, 50)
            else:
                if self.verbose:
                    logger.warning(f"未知的溫度標籤: {temp_filter}")
                return self
            
            # 檢查溫度分布
            temp_min = self.data['temp'].min()
            temp_max = self.data['temp'].max()
            temp_mean = self.data['temp'].mean()
            
            if self.verbose:
                logger.info(f"溫度範圍: {temp_min:.1f} - {temp_max:.1f}°C，平均: {temp_mean:.1f}°C")
            
            # 檢查目標溫度範圍內的數據比例
            in_range = ((self.data['temp'] >= temp_range[0]) & 
                    (self.data['temp'] <= temp_range[1]))
            in_range_count = in_range.sum()
            
            if in_range_count == 0:
                # 如果沒有數據在範圍內，嘗試擴大範圍
                expanded_range = (temp_range[0]-5, temp_range[1]+5)
                expanded_in_range = ((self.data['temp'] >= expanded_range[0]) & 
                                    (self.data['temp'] <= expanded_range[1]))
                expanded_count = expanded_in_range.sum()
                
                if expanded_count > 0:
                    if self.verbose:
                        logger.warning(f"在原始範圍 {temp_range} 內沒有數據，使用擴展範圍 {expanded_range}")
                    temp_range = expanded_range
                    in_range = expanded_in_range
                    in_range_count = expanded_count
                else:
                    if self.verbose:
                        logger.warning(f"溫度過濾 ({temp_filter}): 沒有符合條件的數據，跳過過濾")
                    self.add_metric("temperature_filtered", {
                        "original_count": len(self.data),
                        "filtered_count": len(self.data),
                        "temp_filter": temp_filter,
                        "temp_range": temp_range,
                        "actual_temp_range": [float(temp_min), float(temp_max)],
                        "note": "沒有數據在範圍內，跳過過濾"
                    })
                    return self
            
            in_range_pct = in_range_count / len(self.data) * 100
            if self.verbose:
                logger.info(f"溫度過濾 ({temp_filter}): {in_range_count} 行數據在範圍內 ({in_range_pct:.2f}%)")
            
            # 過濾數據
            original_count = len(self.data)
            self.data = self.data[in_range].reset_index(drop=True)
            
            if self.verbose:
                filtered_count = len(self.data)
                logger.info(f"溫度過濾 ({temp_filter}): 從 {original_count} 行減少到 {filtered_count} 行")
            
            self.add_metric("temperature_filtered", {
                "original_count": original_count,
                "filtered_count": len(self.data),
                "temp_filter": temp_filter,
                "temp_range": list(temp_range),
                "actual_temp_range": [float(temp_min), float(temp_max)]
            })
            
            self.steps_applied.append("filter_by_temperature")
        
        except Exception as e:
            logger.error(f"過濾溫度時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("生成診斷圖")
    def plot_diagnostics(self, output_dir: PathLike) -> 'DataCleaningPipeline':
        """生成診斷圖表
        
        Args:
            output_dir: 輸出目錄
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 設置中文字體支持（如果有的話）
            try:
                plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass
            
            # 檢查是否為空數據框或行數太少
            if len(self.data) < 2:
                logger.warning("數據框為空或行數太少，無法生成診斷圖")
                return self
                
            # 區分充電與放電數據（如果有標記）
            has_charge_discharge = 'is_charging' in self.data.columns
            
            if has_charge_discharge:
                charging_data = self.data[self.data['is_charging']]
                discharging_data = self.data[~self.data['is_charging']]
                
                # 繪製充放電對比圖
                if not charging_data.empty and not discharging_data.empty:
                    # 電流-電壓特性曲線
                    plt.figure(figsize=(12, 8))
                    plt.scatter(charging_data['current'], charging_data['voltage'], 
                              alpha=0.5, label='充電', color='blue', s=3)
                    plt.scatter(discharging_data['current'], discharging_data['voltage'], 
                              alpha=0.5, label='放電', color='red', s=3)
                    plt.xlabel('電流 (A)')
                    plt.ylabel('電壓 (V)')
                    plt.title('電流-電壓特性曲線')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(output_dir / 'current_voltage_characteristic.png', dpi=100)
                    plt.close()
                    
                    # 時間序列對比圖（電流、電壓、溫度）
                    if 'timeindex' in self.data.columns:
                        fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
                        
                        # 電流圖
                        axs[0].scatter(charging_data['timeindex'], charging_data['current'], 
                                    alpha=0.5, label='充電', color='blue', s=2)
                        axs[0].scatter(discharging_data['timeindex'], discharging_data['current'], 
                                    alpha=0.5, label='放電', color='red', s=2)
                        axs[0].set_ylabel('電流 (A)')
                        axs[0].grid(True, alpha=0.3)
                        axs[0].legend()
                        
                        # 電壓圖
                        axs[1].scatter(charging_data['timeindex'], charging_data['voltage'], 
                                    alpha=0.5, label='充電', color='blue', s=2)
                        axs[1].scatter(discharging_data['timeindex'], discharging_data['voltage'], 
                                    alpha=0.5, label='放電', color='red', s=2)
                        axs[1].set_ylabel('電壓 (V)')
                        axs[1].grid(True, alpha=0.3)
                        axs[1].legend()
                        
                        # 溫度圖
                        if 'temp' in self.data.columns:
                            axs[2].scatter(charging_data['timeindex'], charging_data['temp'], 
                                        alpha=0.5, label='充電', color='blue', s=2)
                            axs[2].scatter(discharging_data['timeindex'], discharging_data['temp'], 
                                        alpha=0.5, label='放電', color='red', s=2)
                            axs[2].set_ylabel('溫度 (°C)')
                            axs[2].set_xlabel('時間 (秒)')
                            axs[2].grid(True, alpha=0.3)
                            axs[2].legend()
                        
                        plt.tight_layout()
                        plt.savefig(output_dir / 'time_series_comparison.png', dpi=100)
                        plt.close()
            
            # 數值列的分布圖 (按組繪製避免過多的子圖)
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            # 排除時間相關列和一些特定列
            exclude_cols = ['timeindex', 'index', 'is_charging']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(numeric_cols) > 0:
                # 分組繪製，每組最多12個子圖
                for group_idx in range(0, len(numeric_cols), 12):
                    group_cols = numeric_cols[group_idx:group_idx+12]
                    rows = (len(group_cols) + 2) // 3  # 向上取整
                    plt.figure(figsize=(14, rows * 3))
                    
                    for i, col in enumerate(group_cols):
                        plt.subplot(rows, 3, i+1)
                        try:
                            # 使用更安全的方式繪製直方圖
                            if self.data[col].nunique() > 1:  # 確保數據有變異性
                                self.data[col].hist(bins=min(50, self.data[col].nunique()), alpha=0.7)
                                plt.axvline(self.data[col].mean(), color='r', linestyle='--', alpha=0.7)
                                plt.axvline(self.data[col].median(), color='g', linestyle='-', alpha=0.7)
                                plt.title(f'{col} 分布')
                                plt.grid(True, alpha=0.3)
                            else:
                                plt.text(0.5, 0.5, f"{col} 值全為 {self.data[col].iloc[0]}", 
                                       ha='center', va='center', transform=plt.gca().transAxes)
                                plt.title(f'{col} (無變異)')
                        except Exception as e:
                            logger.warning(f"繪製 {col} 分布圖時出錯: {e}")
                            plt.text(0.5, 0.5, f"繪圖錯誤: {str(e)[:50]}...", 
                                   ha='center', va='center', transform=plt.gca().transAxes)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f'distributions_group{group_idx//12+1}.png', dpi=100)
                    plt.close()
            
            # 時間序列圖
            if 'timeindex' in self.data.columns:
                key_cols = ['voltage', 'current', 'temp', 'bms_cyclecount', 'bms_stateofhealth']
                plot_cols = [c for c in key_cols if c in self.data.columns]
                
                if plot_cols:
                    plt.figure(figsize=(14, len(plot_cols) * 3))
                    
                    for i, col in enumerate(plot_cols):
                        plt.subplot(len(plot_cols), 1, i+1)
                        try:
                            # 確保值不全相同
                            if self.data[col].nunique() > 1:
                                plt.plot(self.data['timeindex'], self.data[col], 'b-', alpha=0.7)
                                plt.title(f'{col} vs. time')
                                plt.ylabel(col)
                                plt.grid(True, alpha=0.3)
                            else:
                                plt.text(0.5, 0.5, f"{col} 值全為 {self.data[col].iloc[0]}", 
                                       ha='center', va='center', transform=plt.gca().transAxes)
                                plt.title(f'{col} (無變異)')
                        except Exception as e:
                            logger.warning(f"繪製 {col} 時間序列圖時出錯: {e}")
                            plt.text(0.5, 0.5, f"繪圖錯誤: {str(e)[:50]}...", 
                                   ha='center', va='center', transform=plt.gca().transAxes)
                    
                    plt.xlabel('time (seconds)')
                    plt.tight_layout()
                    plt.savefig(output_dir / 'time_series.png', dpi=100)
                    plt.close()
            
            # 如果有循環計數，繪製SOH隨循環變化圖
            if 'bms_cyclecount' in self.data.columns and 'bms_stateofhealth' in self.data.columns:
                # 確保數據有足夠的變異性
                if (self.data['bms_cyclecount'].nunique() > 1 and 
                    self.data['bms_stateofhealth'].nunique() > 1):
                    try:
                        plt.figure(figsize=(10, 6))
                        data_for_plot = self.data.groupby('bms_cyclecount')['bms_stateofhealth'].mean().reset_index()
                        plt.plot(data_for_plot['bms_cyclecount'], data_for_plot['bms_stateofhealth'], '-o')
                        plt.title('SOH vs Cycle Count')
                        plt.xlabel('Cycle Count')
                        plt.ylabel('State of Health (%)')
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(output_dir / 'soh_vs_cycle.png', dpi=100)
                        plt.close()
                    except Exception as e:
                        logger.warning(f"繪製 SOH 與循環關係圖時出錯: {e}")
            
            # 電池單元電壓對比圖
            cell_volt_cols = [col for col in self.data.columns if 'cellvolt' in col.lower()]
            if len(cell_volt_cols) > 1:
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # 確保有時間索引
                    if 'timeindex' in self.data.columns:
                        for col in cell_volt_cols:
                            plt.plot(self.data['timeindex'], self.data[col], alpha=0.7, label=col)
                        plt.xlabel('Time (seconds)')
                    else:
                        # 使用數據點索引
                        for col in cell_volt_cols:
                            plt.plot(self.data[col], alpha=0.7, label=col)
                        plt.xlabel('Data Point')
                    
                    plt.ylabel('Cell Voltage (V)')
                    plt.title('Battery Cell Voltages Comparison')
                    plt.grid(True, alpha=0.3)
                    
                    # 如果單元太多，不顯示圖例
                    if len(cell_volt_cols) <= 13:
                        plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / 'cell_voltages.png', dpi=100)
                    plt.close()
                except Exception as e:
                    logger.warning(f"繪製電池單元電壓對比圖時出錯: {e}")
            
            # 相關性熱圖
            try:
                if len(numeric_cols) > 1:
                    corr_cols = [col for col in numeric_cols 
                               if self.data[col].nunique() > 1 and not self.data[col].isna().all()]
                    
                    # 限制相關性分析的列數量
                    if len(corr_cols) > 20:
                        # 選擇主要列（電壓、電流、溫度和關鍵BMS數據）
                        priority_cols = ['voltage', 'current', 'temp', 'bms_rsoc', 'bms_stateofhealth', 
                                       'bms_cyclecount', 'bms_avgcurrent']
                        # 保留優先列，然後加入其他列直到達到20個
                        selected_corr_cols = []
                        for col in priority_cols:
                            if col in corr_cols:
                                selected_corr_cols.append(col)
                        
                        # 添加其他列直到達到20個
                        other_cols = [col for col in corr_cols if col not in selected_corr_cols]
                        remaining_slots = min(20 - len(selected_corr_cols), len(other_cols))
                        if remaining_slots > 0:
                            selected_corr_cols.extend(other_cols[:remaining_slots])
                        
                        corr_cols = selected_corr_cols
                    
                    if len(corr_cols) > 1:
                        plt.figure(figsize=(12, 10))
                        corr = self.data[corr_cols].corr()
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        
                        cmap = plt.cm.viridis
                        
                        # 使用matplotlib直接生成熱圖，不依賴seaborn
                        plt.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
                        plt.colorbar(label='Correlation')
                        
                        # 添加相關係數標籤
                        for i in range(len(corr.columns)):
                            for j in range(len(corr.columns)):
                                if i <= j:  # 只顯示下三角部分
                                    continue
                                plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                                       ha="center", va="center",
                                       color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
                        
                        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
                        plt.yticks(range(len(corr.columns)), corr.columns)
                        plt.title('相關性熱圖')
                        plt.tight_layout()
                        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=100)
                        plt.close()
            except Exception as e:
                logger.warning(f"繪製相關性熱圖時出錯: {e}")
            
            if self.verbose:
                logger.info(f"診斷圖表已保存到: {output_dir}")
            
            self.add_metric("diagnostics_path", str(output_dir))
            self.steps_applied.append("plot_diagnostics")
            
        except Exception as e:
            logger.error(f"繪製診斷圖時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    @log_step("保存數據")
    def save(self, output_path: PathLike) -> 'DataCleaningPipeline':
        """保存清理後的數據
        
        Args:
            output_path: 輸出文件路徑
            
        Returns:
            self: 管道對象，支持鏈式調用
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 處理可能導致 parquet 格式問題的列
            df_to_save = self.data.copy()
            
            for col in df_to_save.columns:
                # 檢查是否有包含字符串的對象列，這些可能導致轉換失敗
                if df_to_save[col].dtype == 'object':
                    try:
                        # 嘗試轉換為數值，如果失敗則轉為字符串
                        df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce')
                    except:
                        # 確保它是字符串類型
                        df_to_save[col] = df_to_save[col].astype(str)
            
            # 根據擴展名決定保存格式
            if output_path.suffix.lower() == '.csv':
                try:
                    df_to_save.to_csv(output_path, index=False, encoding='utf-8')
                    if self.verbose:
                        logger.info(f"數據已保存為 CSV 格式: {output_path}")
                except Exception as e:
                    logger.error(f"保存為 CSV 格式時出錯: {e}，嘗試使用不同編碼")
                    try:
                        df_to_save.to_csv(output_path, index=False, encoding='latin1')
                        if self.verbose:
                            logger.info(f"數據已使用 latin1 編碼保存為 CSV 格式: {output_path}")
                    except Exception as e2:
                        logger.error(f"使用 latin1 編碼保存 CSV 時出錯: {e2}")
                        raise
            elif output_path.suffix.lower() == '.parquet':
                try:
                    # 首先嘗試直接保存
                    df_to_save.to_parquet(output_path, index=False)
                    if self.verbose:
                        logger.info(f"數據已保存為 Parquet 格式: {output_path}")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"保存為 Parquet 格式時出錯: {e}，嘗試修復...")
                    
                    # 修復方案：轉換所有對象類型為字符串
                    for col in df_to_save.columns:
                        if df_to_save[col].dtype == 'object':
                            df_to_save[col] = df_to_save[col].astype(str)
                    
                    try:
                        # 再次嘗試保存
                        df_to_save.to_parquet(output_path, index=False)
                        if self.verbose:
                            logger.info(f"數據已修復並保存為 Parquet 格式: {output_path}")
                    except Exception as e2:
                        if self.verbose:
                            logger.error(f"無法保存為 Parquet 格式: {e2}，將改為保存為CSV格式")
                        # 如果仍然失敗，保存為 CSV
                        csv_path = output_path.with_suffix('.csv')
                        df_to_save.to_csv(csv_path, index=False)
                        output_path = csv_path
                        if self.verbose:
                            logger.info(f"數據已保存為 CSV 格式: {csv_path}")
            else:
                if self.verbose:
                    logger.warning(f"未知的輸出格式: {output_path.suffix}，默認使用 Parquet")
                if output_path.suffix.lower() != '.parquet':
                    output_path = output_path.with_suffix('.parquet')
                
                try:
                    df_to_save.to_parquet(output_path, index=False)
                    if self.verbose:
                        logger.info(f"數據已保存為 Parquet 格式: {output_path}")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"保存為 Parquet 格式時出錯: {e}，改為保存為CSV格式")
                    # 如果失敗，保存為 CSV
                    csv_path = output_path.with_suffix('.csv')
                    df_to_save.to_csv(csv_path, index=False)
                    output_path = csv_path
                    if self.verbose:
                        logger.info(f"數據已保存為 CSV 格式: {csv_path}")
            
            if self.verbose:
                logger.info(f"清理後的數據已保存至: {output_path}")
                logger.info(f"最終數據形狀: {len(self.data)} 行, {len(self.data.columns)} 列")
            
            self.add_metric("output_path", str(output_path))
            self.add_metric("final_shape", [len(self.data), len(self.data.columns)])
            self.steps_applied.append("save")
        
        except Exception as e:
            logger.error(f"保存數據時出錯: {e}")
            if self.verbose:
                logger.error(traceback.format_exc())
        
        return self
    
    def get_summary(self) -> Dict[str, Any]:
        """獲取清理過程的摘要
        
        Returns:
            Dict[str, Any]: 摘要字典
        """
        return {
            "original_shape": [self.original_rows, self.original_cols],
            "final_shape": [len(self.data), len(self.data.columns)],
            "steps_applied": self.steps_applied,
            "metrics": self.metrics
        }


# Modify the main clean_battery_data function to accept the preserve_datetime parameter

def clean_battery_data(input_file: PathLike, output_file: PathLike, temp_filter: Optional[str] = None,
                      plot_diagnostics: bool = False, remove_outliers: bool = True,
                      interpolate_missing: bool = True, verbose: bool = False, 
                      preserve_datetime: bool = False, include_charging: bool = True) -> Optional[pd.DataFrame]:
    """
    清理電池數據並保存結果
    
    Args:
        input_file: 輸入文件路徑
        output_file: 輸出文件路徑
        temp_filter: 溫度過濾條件 (例如 '5deg', '25deg', '45deg')
        plot_diagnostics: 是否繪製診斷圖
        remove_outliers: 是否移除異常值
        interpolate_missing: 是否插補缺失值
        verbose: 是否打印詳細日誌
        preserve_datetime: 是否保留原始 DateTime 欄位
        include_charging: 是否包含充電數據，False 則僅保留放電數據
    
    Returns:
        pd.DataFrame: 清理後的數據框，如果失敗則返回 None
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    
    logger.info(f"開始清理文件: {input_file}")
    if temp_filter:
        logger.info(f"應用溫度過濾: {temp_filter}")
    if preserve_datetime:
        logger.info(f"將保留原始 DateTime 欄位")
    if not include_charging:
        logger.info(f"僅保留放電數據")
    
    # 讀取數據
    try:
        if input_file.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(input_file, low_memory=False)
            except Exception as e:
                logger.error(f"讀取CSV文件時出錯: {e}")
                for encoding in ['latin1', 'cp1252', 'utf-8-sig']:
                    try:
                        df = pd.read_csv(input_file, encoding=encoding)
                        logger.info(f"使用 {encoding} 編碼成功讀取CSV")
                        break
                    except:
                        continue
                else:
                    logger.error(f"無法讀取文件 {input_file}")
                    return None
        elif input_file.suffix.lower() == '.parquet':
            df = pd.read_parquet(input_file)
        elif input_file.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
        else:
            logger.error(f"不支持的文件格式: {input_file.suffix}")
            return None
    except Exception as e:
        logger.error(f"讀取文件時出錯: {e}")
        logger.error(traceback.format_exc())
        return None
    
    # 檢查是否有 DateTime 列，並確保它被保留
    has_datetime = 'DateTime' in df.columns
    
    if has_datetime and preserve_datetime:
        logger.info(f"檢測到 DateTime 列，將在整個處理過程中保留")
        # 創建 DateTime 的備份
        df['original_DateTime'] = df['DateTime']
    
    # 提取文件溫度
    temp_str, _ = extract_temp_and_channel(input_file)
    temp_str_lower = temp_str.lower() if temp_str else None
    file_temp = None
    if temp_str_lower == '5degree':
        file_temp = '5deg'
    elif temp_str_lower == '25degree':
        file_temp = '25deg'
    elif temp_str_lower == '45degree':
        file_temp = '45deg'
    
    # 檢查溫度是否符合過濾條件
    if temp_filter and file_temp != temp_filter.lower():
        logger.info(f"文件 {input_file} 的溫度 ({temp_str}) 不符合過濾條件 ({temp_filter})，跳過處理")
        return None
    
    # 初始化數據清理管道
    pipeline = DataCleaningPipeline(data=df, verbose=verbose)
    
    try:
        # 按順序執行清理步驟
        pipeline = pipeline.preprocess()
        pipeline = pipeline.unify_columns()
        
        # 處理時間戳
        pipeline = pipeline.handle_timestamps()
        
        # 標記充電和放電
        pipeline = pipeline.mark_charge_discharge()
        
        # 選擇是否過濾充電數據
        if not include_charging:
            pipeline = pipeline.filter_charging_data(keep_charging=False)
            
        pipeline = pipeline.check_missing_values()
        
        # 可選步驟：插補缺失值
        if interpolate_missing:
            pipeline = pipeline.interpolate_missing()
        
        # 單位轉換
        pipeline = pipeline.convert_units()
        
        # 對單元電壓列進行趨勢校正
        # 首先檢查存在的列
        available_cell_volt_columns = []
        for i in range(1, 14):
            # 檢查不同可能的列名
            possible_names = [
                f'BMS_CellVolt{i}',
                f'bms_cellvolt{i}',
                f'cellvolt{i}'
            ]
        
            # 使用大小寫不敏感匹配
            found = False
            for name in possible_names:
                matching_cols = [col for col in pipeline.data.columns if col.lower() == name.lower()]
                if matching_cols:
                    available_cell_volt_columns.append(matching_cols[0])
                    if verbose:
                        logger.info(f"找到電池單元電壓列: {matching_cols[0]}")
                    found = True
                    break
            
            if not found and verbose:
                logger.info(f"未找到電池單元電壓 {i} 的列")

        if available_cell_volt_columns:
            pipeline = pipeline.detrend_time_series(columns=available_cell_volt_columns, window=60)

            # 將處理後的偏差加回移動平均，恢復原始列
            for col in available_cell_volt_columns:
                detrended_col = f'{col}_detrended'
                if detrended_col in pipeline.data.columns:
                    try:
                        moving_avg = pipeline.data[col].rolling(window=60, min_periods=1, center=True).mean()
                        pipeline.data[col] = moving_avg + pipeline.data[detrended_col]
                        # 移除臨時的 detrended 列
                        pipeline.data = pipeline.data.drop(columns=[detrended_col])
                    except Exception as e:
                        logger.error(f"還原 {col} 的趨勢時出錯: {e}")
                        # 如果出錯，確保 detrended 列被移除
                        if detrended_col in pipeline.data.columns:
                            pipeline.data = pipeline.data.drop(columns=[detrended_col])
        else:
            logger.info("沒有找到電池單元電壓列，跳過趨勢校正")
        
        # 處理異常值，考慮充放電狀態
        if remove_outliers:
            # 僅針對存在的列設置閾值
            column_thresholds = {'BMS_PackCurrent': 3.0, 'BMS_AvgCurrent': 3.0, 'current': 3.0}
            
            # 檢查並添加 detrended 列的閾值
            for col in available_cell_volt_columns:
                detrended_col = f'{col}_detrended'
                if detrended_col in pipeline.data.columns:
                    column_thresholds[detrended_col] = 1.5
            
            pipeline = pipeline.handle_outliers(
                method='iqr',
                threshold=1.5,
                column_thresholds=column_thresholds,
                respect_charge_discharge=True
            )
        
        # 檢查物理約束
        pipeline = pipeline.check_physical_constraints()
        pipeline = pipeline.fix_time_series()
        pipeline = pipeline.convert_dtypes()
        
        # 溫度過濾（如果提供了 temp_filter）
        if temp_filter:
            pipeline = pipeline.filter_by_temperature(temp_filter)
        
        # 診斷圖（如果啟用）
        if plot_diagnostics:
            diagnostics_dir = output_file.parent / 'diagnostics'
            pipeline = pipeline.plot_diagnostics(diagnostics_dir)
        
        # 如果需要保留 DateTime 列，且清理過程中丟失了它，則從備份中恢復
        if preserve_datetime and has_datetime:
            if 'original_DateTime' in pipeline.data.columns and 'DateTime' not in pipeline.data.columns:
                pipeline.data['DateTime'] = pipeline.data['original_DateTime']
                pipeline.data = pipeline.data.drop(columns=['original_DateTime'])
                logger.info(f"已從備份中恢復 DateTime 列")
            elif 'DateTime' not in pipeline.data.columns:
                logger.warning(f"無法恢復 DateTime 列，因為備份也不存在")
        
        # 保存結果
        pipeline = pipeline.save(output_file)
        
        logger.info(f"數據清理完成，結果已保存至: {output_file}")
        # 確認 DateTime 列是否被成功保留
        if preserve_datetime and 'DateTime' in pipeline.data.columns:
            logger.info(f"DateTime 列已成功保留在輸出文件中")
        
        return pipeline.data
    
    except Exception as e:
        logger.error(f"數據清理過程中出錯: {e}")
        logger.error(traceback.format_exc())
        return None
    
def is_valid_cycle_column(df: pd.DataFrame, column_name: str) -> bool:
    """
    檢查列是否是有效的循環計數列
    
    Args:
        df: 數據框
        column_name: 列名
    
    Returns:
        bool: 是否是有效的循環計數列
    """
    if column_name not in df.columns:
        return False
    
    try:
        # 嘗試將欄位轉為數值，無效值轉為 NaN
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        # 檢查是否有有效數據
        if df[column_name].isna().all():
            logger.warning(f"欄位 '{column_name}' 全為 NaN")
            return False
        
        # 檢查範圍是否合理
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        if min_val < 0 or max_val > 10000:
            logger.warning(f"欄位 '{column_name}' 的值範圍無效: {min_val} - {max_val}")
            return False
        
        return True
    except Exception as e:
        logger.warning(f"檢查欄位 '{column_name}' 時出錯: {e}")
        return False

def extract_cycle_range(df: pd.DataFrame, file_path: Path) -> str:
    """
    從數據框或文件名中提取循環範圍
    
    Args:
        df: 數據框
        file_path: 檔案路徑
    
    Returns:
        str: 循環範圍，格式為 "xxx-xxx"
    """
    cycle_range = "000-999"  # 默認值
    cycle_column_names = [
        'bms_cyclecount', 'BMS_CycleCount', 'cycle', 'cycle_count', 'cyclecount', 'cycles', 'cycle_number'
    ]
    
    valid_cycle_column = None
    # 將 DataFrame 的欄位名稱轉為小寫進行比較
    df_columns_lower = [col.lower() for col in df.columns]
    for col in cycle_column_names:
        if col.lower() in df_columns_lower:
            # 找到原始欄位名稱
            valid_cycle_column = df.columns[df_columns_lower.index(col.lower())]
            if is_valid_cycle_column(df, valid_cycle_column):
                break
            else:
                valid_cycle_column = None
    
    # 如果沒有找到，嘗試模糊匹配包含 'cycle' 或 'count' 的欄位
    if valid_cycle_column is None:
        for col in df.columns:
            if ('cycle' in col.lower() or 'count' in col.lower()) and is_valid_cycle_column(df, col):
                valid_cycle_column = col
                break
    
    # 從數據中提取循環範圍
    if valid_cycle_column is not None:
        try:
            df[valid_cycle_column] = pd.to_numeric(df[valid_cycle_column], errors='coerce')
            if not df[valid_cycle_column].isna().all():
                min_cycle = int(df[valid_cycle_column].min())
                max_cycle = int(df[valid_cycle_column].max())
                cycle_range = f"{min_cycle:03d}-{max_cycle:03d}"
                logger.info(f"從欄位 '{valid_cycle_column}' 提取循環範圍: {cycle_range}")
                return cycle_range
            else:
                logger.warning(f"欄位 '{valid_cycle_column}' 全為 NaN")
        except Exception as e:
            logger.warning(f"從 '{valid_cycle_column}' 提取循環範圍失敗: {e}")
    
    # 如果數據中沒有循環計數，嘗試從文件名提取
    logger.warning(f"無法從數據中提取循環範圍，嘗試從文件名提取: {file_path.name}")
    try:
        filename = file_path.stem
        # 嘗試匹配 xxx-xxx 或 xxx_xxx 格式
        cycle_match = re.search(r'(?:cycle)?(\d{3})[-_](\d{3})', filename, re.IGNORECASE)
        if cycle_match:
            start = int(cycle_match.group(1))
            end = int(cycle_match.group(2))
            cycle_range = f"{start:03d}-{end:03d}"
            logger.info(f"從文件名提取循環範圍: {cycle_range}")
            return cycle_range
        
        # 嘗試匹配 6 位數字（例如 003005）
        matches = re.findall(r'(\d{6})', filename)
        if matches and len(matches[0]) == 6:
            start = int(matches[0][:3])
            end = int(matches[0][3:])
            cycle_range = f"{start:03d}-{end:03d}"
            logger.info(f"從文件名提取 6 位數字循環範圍: {cycle_range}")
            return cycle_range
        
        # 嘗試匹配兩個三位數
        numbers = re.findall(r'\d{3}', filename)
        if len(numbers) >= 2:
            start = int(numbers[0])
            end = int(numbers[1])
            cycle_range = f"{start:03d}-{end:03d}"
            logger.info(f"從文件名提取兩個三位數作為循環範圍: {cycle_range}")
            return cycle_range
    except Exception as e:
        logger.warning(f"從文件名提取循環範圍失敗: {e}")
    
    # 如果都失敗，使用時間戳作為默認值
    timestamp = int(time.time())
    base = timestamp % 1000
    cycle_range = f"{base:03d}-{base:03d}"
    logger.warning(f"無法提取循環範圍，使用時間戳生成默認值: {cycle_range}")
    return cycle_range

def rename_by_cycle_count(file_path: PathLike, temp: str = None, channel: str = None) -> str:
    """
    根據循環計數重命名檔案
    
    Args:
        file_path: 檔案路徑
        temp: 溫度 (5degree, 25degree, 45degree)，如果為 None 則從路徑提取
        channel: 通道 (Ch1, Ch2, Ch3, Ch4)，如果為 None 則從路徑提取
    
    Returns:
        str: 新檔案路徑
    """
    file_path = Path(file_path)
    
    # 文件不存在時返回原路徑
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return str(file_path)
    
    try:
        # 讀取數據
        if file_path.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except Exception as e:
                logger.error(f"讀取CSV文件時出錯: {e}")
                for encoding in ['latin1', 'cp1252', 'utf-8-sig']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        logger.info(f"使用 {encoding} 編碼成功讀取CSV")
                        break
                    except:
                        continue
                else:
                    logger.error(f"無法讀取文件 {file_path}")
                    return str(file_path)
        elif file_path.suffix.lower() == '.parquet':
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                logger.error(f"讀取Parquet文件時出錯: {e}")
                return str(file_path)
        else:
            logger.error(f"不支持的文件格式: {file_path.suffix}")
            return str(file_path)
        
        # 如果未提供溫度和通道，則從路徑提取（作為後備）
        # 但確保不會覆蓋明確提供的參數
        if temp is None or channel is None:
            extracted_temp, extracted_channel = extract_temp_and_channel(file_path)
            if temp is None:
                temp = extracted_temp
            if channel is None:
                channel = extracted_channel
        
        # 確認溫度格式是否正確（避免從25degree誤判為5degree）
        # 如果文件名中包含明確的溫度標識，則優先使用
        original_filename = file_path.stem
        if '25degree' in original_filename.lower() and temp == '5degree':
            logger.warning(f"檢測到溫度標識不一致: 文件名包含'25degree'但被檢測為'{temp}'，修正為'25degree'")
            temp = '25degree'
        elif '45degree' in original_filename.lower() and temp != '45degree':
            logger.warning(f"檢測到溫度標識不一致: 文件名包含'45degree'但被檢測為'{temp}'，修正為'45degree'")
            temp = '45degree'
        elif '5degree' in original_filename.lower() and temp != '5degree' and '25degree' not in original_filename.lower() and '45degree' not in original_filename.lower():
            logger.warning(f"檢測到溫度標識不一致: 文件名包含'5degree'但被檢測為'{temp}'，修正為'5degree'")
            temp = '5degree'
        
        # 提取循環範圍
        cycle_range = extract_cycle_range(df, file_path)
        
        # 溫度簡稱 - 從完整的溫度字符串中提取數字部分
        if temp.lower().startswith('5'):
            temp_short = '5'
        elif temp.lower().startswith('25'):
            temp_short = '25'
        elif temp.lower().startswith('45'):
            temp_short = '45'
        else:
            temp_short = 'X'
        
        # 構建新文件名 - 使用正確的溫度簡稱
        base_new_name = f"{temp_short}_{channel}_{cycle_range}_cleaned"
        new_name = f"{base_new_name}{file_path.suffix}"
        new_path = file_path.parent / new_name
        
        # 處理檔名衝突
        counter = 1
        while new_path.exists() and new_path != file_path:
            new_name = f"{base_new_name}_{counter}{file_path.suffix}"
            new_path = file_path.parent / new_name
            counter += 1
        
        # 重命名檔案
        if new_path != file_path:
            try:
                file_path.rename(new_path)
                logger.info(f"已重命名: {file_path.name} -> {new_name}")
            except OSError as e:
                logger.error(f"重命名文件時出錯: {e}")
                # 嘗試使用複製再刪除的方式
                import shutil
                shutil.copy2(file_path, new_path)
                file_path.unlink()
                logger.info(f"已通過複製方式重命名: {file_path.name} -> {new_name}")
        else:
            logger.info(f"文件名已符合規範，無需重命名: {file_path.name}")
        
        return str(new_path)
    
    except Exception as e:
        logger.error(f"重命名文件時出錯: {e}")
        logger.error(traceback.format_exc())
        return str(file_path)

def batch_process_directory(input_dir: PathLike, output_dir: PathLike,
                           temp_filter: Optional[str] = None,
                           plot_diagnostics: bool = False,
                           remove_outliers: bool = True,
                           interpolate_missing: bool = True,
                           include_charging: bool = True,
                           max_workers: int = 4) -> Tuple[int, int]:
    """
    批量處理目錄中的電池數據

    Args:
        input_dir: 輸入目錄
        output_dir: 輸出目錄
        temp_filter: 溫度過濾，如 '5deg', '25deg', '45deg'
        plot_diagnostics: 是否生成診斷圖
        remove_outliers: 是否移除異常值
        interpolate_missing: 是否插補缺失值
        include_charging: 是否包含充電數據
        max_workers: 最大並行處理數量

    Returns:
        Tuple[int, int]: 成功和失敗數量
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 計算總文件數並過濾符合溫度的文件
    all_files = []
    temp_filter_str = temp_filter.lower() if temp_filter else None
    for ext in ['*.csv', '*.parquet', '*.xlsx', '*.xls']:
        for file_path in input_dir.glob(f"**/{ext}"):
            # 提取文件溫度
            temp_str, _ = extract_temp_and_channel(file_path)
            temp_str_lower = temp_str.lower() if temp_str else None
            file_temp = None
            if temp_str_lower == '5degree' or temp_str_lower == '5degre':
                file_temp = '5deg'
            elif temp_str_lower == '25degree' or temp_str_lower == '25degre':
                file_temp = '25deg'
            elif temp_str_lower == '45degree' or temp_str_lower == '45degre':
                file_temp = '45deg'

            # 溫度過濾
            if temp_filter_str:
                if file_temp != temp_filter_str:
                    logger.info(f"跳過文件 {file_path}，因為其溫度 ({file_temp}) 不符合過濾條件 ({temp_filter_str})")
                    continue
            all_files.append(file_path)

    total_files = len(all_files)
    logger.info(f"共找到 {total_files} 個符合條件的文件需要處理")

    # 處理計數器
    processed = 0
    success = 0
    failed = 0
    start_time = time.time()

    # 檢查是否使用並行處理
    if max_workers > 1 and total_files > 1:
        # 定義處理單個文件的函數
        def process_file(file_path):
            nonlocal processed, success, failed
            try:
                rel_path = file_path.relative_to(input_dir)
                output_path = output_dir / rel_path.with_suffix('.parquet')
                output_path.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"處理文件: {file_path}")
                cleaned_data = clean_battery_data(
                    input_file=file_path,
                    output_file=output_path,
                    temp_filter=temp_filter,
                    plot_diagnostics=plot_diagnostics,
                    remove_outliers=remove_outliers,
                    interpolate_missing=interpolate_missing,
                    include_charging=include_charging
                )

                if cleaned_data is not None:
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"處理文件 {file_path} 時出錯: {e}")
                return False

        # 使用 ThreadPoolExecutor 並行處理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任務並獲取未來結果
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in all_files}
            
            # 處理結果
            for future in future_to_file:
                file_path = future_to_file[future]
                processed += 1
                try:
                    result = future.result()
                    if result:
                        success += 1
                    else:
                        failed += 1
                        FAILED_FILES.append(str(file_path))
                except Exception as e:
                    logger.error(f"獲取處理結果時出錯: {e}")
                    failed += 1
                    FAILED_FILES.append(str(file_path))

                # 打印進度
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / processed
                remaining_files = total_files - processed
                est_remaining_time = avg_time_per_file * remaining_files

                logger.info(f"進度: {processed}/{total_files} 檔案 "
                          f"({processed/total_files*100:.2f}%), "
                          f"成功: {success}, 失敗: {failed}, "
                          f"預計剩餘時間: {timedelta(seconds=int(est_remaining_time))}")
    else:
        # 串行處理
        for file_path in all_files:
            processed += 1
            rel_path = file_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.parquet')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"處理文件 {processed}/{total_files}: {file_path}")
            cleaned_data = clean_battery_data(
                input_file=file_path,
                output_file=output_path,
                temp_filter=temp_filter,
                plot_diagnostics=plot_diagnostics,
                remove_outliers=remove_outliers,
                interpolate_missing=interpolate_missing,
                include_charging=include_charging
            )

            if cleaned_data is not None:
                success += 1
            else:
                failed += 1
                FAILED_FILES.append(str(file_path))

            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / processed
            remaining_files = total_files - processed
            est_remaining_time = avg_time_per_file * remaining_files

            logger.info(f"進度: {processed}/{total_files} 檔案 "
                      f"({processed/total_files*100:.2f}%), "
                      f"成功: {success}, 失敗: {failed}, "
                      f"預計剩餘時間: {timedelta(seconds=int(est_remaining_time))}")

    if CLEANING_REPORT:
        report_path = output_dir / 'cleaning_report.csv'
        pd.DataFrame(CLEANING_REPORT).to_csv(report_path, index=False)
        logger.info(f"清理報告已保存至: {report_path}")

    if FAILED_FILES:
        failed_path = output_dir / 'failed_files.txt'
        with open(failed_path, 'w', encoding='utf-8') as f:
            for path in FAILED_FILES:
                f.write(f"{path}\n")
        logger.info(f"失敗文件列表已保存至: {failed_path}")

    total_time = time.time() - start_time
    logger.info(f"批處理完成，共處理 {total_files} 個文件，"
              f"成功: {success}，失敗: {failed}，"
              f"總耗時: {timedelta(seconds=int(total_time))}")

    return success, failed


def main():
    parser = argparse.ArgumentParser(description="電池數據清理與處理系統")
    parser.add_argument("--input", type=str, required=True, help="輸入數據文件或目錄路徑")
    parser.add_argument("--output", type=str, default=None, help="輸出數據文件或目錄路徑")
    parser.add_argument("--batch", action="store_true", help="批量處理模式（輸入為目錄）")
    parser.add_argument("--temp", type=str, default=None, choices=["5deg", "25deg", "45deg"], help="按溫度過濾數據")
    parser.add_argument("--no-outlier-removal", action="store_false", dest="remove_outliers", help="禁用異常值移除")
    parser.add_argument("--no-interpolation", action="store_false", dest="interpolate_missing", help="禁用缺失值插補")
    parser.add_argument("--plot-diagnostics", action="store_true", help="繪製診斷圖")
    parser.add_argument("--rename", action="store_true", help="根據循環範圍重命名輸出文件")
    parser.add_argument("--channel", type=str, default=None, help="通道標識（配合--rename使用，例如 Ch1）")
    parser.add_argument("--temp-str", type=str, default=None, help="溫度字符串（配合--rename使用，例如 5degree）")
    parser.add_argument("--preserve-datetime", action="store_true", help="保留原始 DateTime 列")
    parser.add_argument("--no-charging", action="store_false", dest="include_charging", help="排除充電數據")
    parser.add_argument("--workers", type=int, default=4, help="並行處理的工作線程數")
    parser.add_argument("--quiet", action="store_false", dest="verbose", help="靜默模式")
    args = parser.parse_args()

    try:
        # 移除路徑中的多餘引號
        input_path_str = args.input.strip('"')
        input_path = Path(input_path_str)
        logger.info(f"輸入路徑: {input_path}")

        # 批處理模式
        if args.batch or input_path.is_dir():
            if args.output is None:
                output_path = input_path.parent / 'cleaned_data'
            else:
                output_path = Path(args.output.strip('"'))
            success, failed = batch_process_directory(
                input_dir=input_path,
                output_dir=output_path,
                temp_filter=args.temp,
                plot_diagnostics=args.plot_diagnostics,
                remove_outliers=args.remove_outliers,
                interpolate_missing=args.interpolate_missing,
                include_charging=args.include_charging,
                max_workers=args.workers
            )
            logger.info(f"批處理完成，成功: {success}，失敗: {failed}")
        # 單文件模式
        else:
            if args.output is None:
                output_path = input_path.parent / f"{input_path.stem}_cleaned.parquet"
            else:
                output_path = Path(args.output.strip('"'))
            cleaned_data = clean_battery_data(
                input_file=input_path,
                output_file=output_path,
                temp_filter=args.temp,
                plot_diagnostics=args.plot_diagnostics,
                remove_outliers=args.remove_outliers,
                interpolate_missing=args.interpolate_missing,
                verbose=args.verbose,
                preserve_datetime=args.preserve_datetime,
                include_charging=args.include_charging
            )
            if args.rename and cleaned_data is not None and args.channel and args.temp_str:
                rename_by_cycle_count(output_path, args.temp_str, args.channel)
            print(f"數據清理完成！")
            if cleaned_data is not None:
                original_df = None
                try:
                    if input_path.suffix.lower() == '.csv':
                        original_df = pd.read_csv(input_path)
                    elif input_path.suffix.lower() == '.parquet':
                        original_df = pd.read_parquet(input_path)
                    elif input_path.suffix.lower() in ['.xlsx', '.xls']:
                        original_df = pd.read_excel(input_path)
                except:
                    pass
                print(f"  - 原始行數: {len(original_df) if original_df is not None else '未知'}")
                print(f"  - 清理後行數: {len(cleaned_data)}")
                print(f"  - 清理後列數: {cleaned_data.shape[1]}")
                print(f"  - 已保存至: {output_path}")
        return 0
    except Exception as e:
        logger.error(f"程序執行失敗: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        if HAS_CUSTOM_LOGGING:
            pass # Add pass to fix indentation error after removing shutdown_logging

if __name__ == "__main__":
    sys.exit(main())
